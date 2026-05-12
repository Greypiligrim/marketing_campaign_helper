"""Веб-сервис маркетингового бота: REST API + простой веб-чат.

Запуск:
    uvicorn web.app:app --reload
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core import service
from db.models import create_tables, init_default_templates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# Как часто подчищаем просроченные «ожидающие подтверждения» кампании.
_PENDING_CLEANUP_INTERVAL_SEC = 3600


async def _pending_cleanup_loop() -> None:
    while True:
        try:
            removed = await service.cleanup_pending()
            if removed:
                logger.info(f"Удалено просроченных pending-кампаний: {removed}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"pending cleanup error: {e}", exc_info=True)
        await asyncio.sleep(_PENDING_CLEANUP_INTERVAL_SEC)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_tables()
    await init_default_templates()
    await service.cleanup_pending()
    task = asyncio.create_task(_pending_cleanup_loop())
    logger.info("🌐 Веб-сервис запущен")
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="Маркетинговый сервис", version="1.0.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Схемы
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    user_name: Optional[str] = Field(default=None, max_length=100)
    user_role: Optional[str] = Field(default=None, max_length=100)


class ChatResponse(BaseModel):
    type: str  # "question" | "other" | "campaign" | "clarify"
    answer: Optional[str] = None
    pending_id: Optional[str] = None
    preview: Optional[Dict[str, Any]] = None
    missing: Optional[List[str]] = None


class ConfirmRequest(BaseModel):
    pending_id: str
    user_name: Optional[str] = Field(default=None, max_length=100)
    user_role: Optional[str] = Field(default=None, max_length=100)


class RefineRequest(BaseModel):
    pending_id: str
    message: str = Field(..., min_length=1, max_length=4000)
    user_name: Optional[str] = Field(default=None, max_length=100)
    user_role: Optional[str] = Field(default=None, max_length=100)


def _format_user(name: Optional[str], role: Optional[str]) -> Optional[str]:
    name = (name or "").strip()
    role = (role or "").strip()
    if name and role:
        return f"{name} ({role})"
    return name or role or None


class TemplateSubtaskIn(BaseModel):
    name: str
    responsible_role: str
    budget_percentage: float
    execution_days_ratio: float


class TemplateIn(BaseModel):
    name: str
    description: str = ""
    total_budget_rub: float
    participants_example: str = ""
    deadline_example: str = ""
    subtasks: List[TemplateSubtaskIn]


class UserIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    role: Optional[str] = Field(default=None, max_length=100)


async def _register_user_safely(name: Optional[str], role: Optional[str]) -> None:
    """Регистрирует участника чата, не роняя запрос при ошибке."""
    if not (name or "").strip():
        return
    try:
        await service.register_user(name, role)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"register_user failed: {e}")


# --------------------------------------------------------------------------- #
# Чат-эндпоинт (основной интерфейс вместо Telegram)
# --------------------------------------------------------------------------- #
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    text = req.message.strip()
    await _register_user_safely(req.user_name, req.user_role)
    try:
        msg_type = await service.classify_message(text)
    except Exception as e:
        logger.error(f"classify error: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail="Ошибка обработки сообщения.")

    if msg_type == "question":
        try:
            answer = await service.answer_question(text)
        except Exception as e:
            logger.error(f"answer error: {e}", exc_info=True)
            raise HTTPException(status_code=502, detail="Не удалось ответить на вопрос.")
        return ChatResponse(type="question", answer=answer)

    if msg_type == "other":
        return ChatResponse(
            type="other",
            answer=(
                "👋 Я сервис для управления маркетинговыми кампаниями.\n"
                "Отправьте описание кампании (название, задача, бюджет, дедлайн, участники) "
                "или задайте вопрос о существующих кампаниях."
            ),
        )

    # campaign
    try:
        parsed = await service.parse_campaign(text, user_display=_format_user(req.user_name, req.user_role))
    except Exception as e:
        logger.error(f"parse campaign error: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail="Не удалось распознать кампанию. Проверьте формат.")

    parsed = service.enrich_parsed(parsed)
    pending_id = await service.create_pending(parsed)
    missing = service.missing_campaign_fields(parsed)
    if missing:
        return ChatResponse(
            type="clarify",
            pending_id=pending_id,
            preview=parsed,
            missing=missing,
            answer=(
                "Почти готово 👍 Чтобы создать кампанию, не хватает: "
                + ", ".join(missing)
                + ".\nОтправьте недостающие данные сообщением — я их добавлю."
            ),
        )
    return ChatResponse(type="campaign", pending_id=pending_id, preview=parsed)


@app.post("/api/campaigns/confirm")
async def confirm_campaign(req: ConfirmRequest) -> Dict[str, Any]:
    parsed = await service.get_pending(req.pending_id)
    if parsed is None:
        raise HTTPException(status_code=404, detail="Данные кампании не найдены или устарели.")
    try:
        campaign = await service.create_campaign_from_parsed(
            parsed, created_by=_format_user(req.user_name, req.user_role)
        )
    except service.CampaignError as e:
        # Оставляем pending на месте — пользователь может исправить и повторить.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"create campaign error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка сохранения кампании.")
    await service.delete_pending(req.pending_id)
    return {"campaign": campaign}


@app.post("/api/campaigns/refine", response_model=ChatResponse)
async def refine_campaign(req: RefineRequest) -> ChatResponse:
    """Дозаполняет ожидающую кампанию данными из нового сообщения пользователя.

    Возвращает такой же ChatResponse, как /api/chat: type="campaign", когда данных
    достаточно, или снова type="clarify" со списком всё ещё недостающих полей.
    """
    base = await service.get_pending(req.pending_id)
    if base is None:
        raise HTTPException(status_code=404, detail="Данные кампании не найдены или устарели.")

    try:
        extra = await service.parse_campaign(req.message, user_display=_format_user(req.user_name, req.user_role))
    except Exception as e:
        logger.error(f"refine parse error: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail="Не удалось распознать данные. Попробуйте сформулировать иначе.")

    merged = service.enrich_parsed(service.merge_parsed(base, extra))
    await service.update_pending(req.pending_id, merged)

    missing = service.missing_campaign_fields(merged)
    if missing:
        return ChatResponse(
            type="clarify",
            pending_id=req.pending_id,
            preview=merged,
            missing=missing,
            answer="Записал. Осталось уточнить: " + ", ".join(missing) + ".",
        )
    return ChatResponse(type="campaign", pending_id=req.pending_id, preview=merged)


@app.post("/api/campaigns/cancel")
async def cancel_campaign(req: ConfirmRequest) -> Dict[str, str]:
    await service.delete_pending(req.pending_id)
    return {"status": "cancelled"}


# --------------------------------------------------------------------------- #
# Кампании
# --------------------------------------------------------------------------- #
@app.get("/api/campaigns")
async def get_campaigns() -> List[Dict[str, Any]]:
    return await service.list_campaigns()


@app.delete("/api/campaigns/{name}")
async def remove_campaign(name: str) -> Dict[str, str]:
    deleted = await service.delete_campaign(name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Кампания '{name}' не найдена.")
    return {"status": "deleted"}


# --------------------------------------------------------------------------- #
# Шаблоны
# --------------------------------------------------------------------------- #
@app.get("/api/templates")
async def get_templates() -> List[Dict[str, Any]]:
    return await service.list_templates()


@app.get("/api/templates/{name}")
async def get_template(name: str) -> Dict[str, Any]:
    template = await service.get_template(name)
    if template is None:
        raise HTTPException(status_code=404, detail=f"Шаблон '{name}' не найден.")
    return template


@app.post("/api/templates")
async def create_template(req: TemplateIn) -> Dict[str, Any]:
    try:
        return await service.add_template(
            name=req.name,
            description=req.description,
            total_budget_rub=req.total_budget_rub,
            participants_example=req.participants_example,
            deadline_example=req.deadline_example,
            subtasks=[st.model_dump() for st in req.subtasks],
        )
    except service.CampaignError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/templates/{name}")
async def edit_template(name: str, req: TemplateIn) -> Dict[str, Any]:
    """Редактирует существующий шаблон. `req.name` — это новое имя (может совпадать)."""
    try:
        return await service.update_template(
            name=name,
            new_name=req.name,
            description=req.description,
            total_budget_rub=req.total_budget_rub,
            participants_example=req.participants_example,
            deadline_example=req.deadline_example,
            subtasks=[st.model_dump() for st in req.subtasks],
        )
    except service.CampaignError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/templates/{name}")
async def remove_template(name: str) -> Dict[str, str]:
    deleted = await service.delete_template(name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Шаблон '{name}' не найден.")
    return {"status": "deleted"}


# --------------------------------------------------------------------------- #
# Участники чата
# --------------------------------------------------------------------------- #
@app.get("/api/users")
async def get_users() -> List[Dict[str, Any]]:
    return await service.list_users()


@app.post("/api/users")
async def register_user(req: UserIn) -> Dict[str, Any]:
    try:
        return await service.register_user(req.name, req.role)
    except service.CampaignError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/users/{name}")
async def remove_user(name: str) -> Dict[str, str]:
    deleted = await service.delete_user(name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Участник '{name}' не найден.")
    return {"status": "deleted"}


# --------------------------------------------------------------------------- #
# Напоминания (вместо job_queue) — что нужно сделать сегодня
# --------------------------------------------------------------------------- #
@app.get("/api/reminders/today")
async def reminders_today() -> List[Dict[str, Any]]:
    return await service.subtasks_due_today()


# --------------------------------------------------------------------------- #
# Служебное
# --------------------------------------------------------------------------- #
@app.get("/api/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


# --------------------------------------------------------------------------- #
# Веб-чат (статика)
# --------------------------------------------------------------------------- #
@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
