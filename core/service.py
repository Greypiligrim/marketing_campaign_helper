"""Бизнес-логика маркетингового сервиса без привязки к Telegram.

Этот модуль содержит чистые async-функции, которые раньше жили в обработчиках
бота. Веб-слой (FastAPI) вызывает их напрямую.
"""
import json
import logging
import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from dateparser import parse as parse_date
from sqlalchemy import and_, delete as sa_delete, select
from sqlalchemy.orm import selectinload

from db.models import (
    AsyncSessionLocal,
    Campaign,
    CampaignTemplate,
    ChatUser,
    PendingCampaign,
    Subtask,
    TemplateSubtask,
)
from ml.llm import NemotronLLM

logger = logging.getLogger(__name__)

llm_client = NemotronLLM()

# Сколько живёт распарсенная, но не подтверждённая кампания, прежде чем её уберёт
# фоновая очистка (часов).
PENDING_TTL_HOURS = 24


# --------------------------------------------------------------------------- #
# Вспомогательные функции
# --------------------------------------------------------------------------- #
def parse_date_to_datetime(date_str: Optional[str]) -> Optional[datetime]:
    """Парсит дату из строки в datetime (ISO или русские форматы)."""
    if not date_str:
        return None

    if isinstance(date_str, str) and len(date_str) == 10 and date_str[4] == "-" and date_str[7] == "-":
        try:
            return datetime.fromisoformat(date_str)
        except ValueError:
            pass

    dt = parse_date(date_str, languages=["ru", "en"], settings={"DATE_ORDER": "DMY"})
    return dt if dt else None


def campaign_to_dict(c: Campaign) -> Dict[str, Any]:
    return {
        "id": c.id,
        "name": c.name,
        "responsible": c.responsible,
        "deadline": c.deadline.date().isoformat() if c.deadline else None,
        "budget_rub": c.budget_rub,
        "created_by": c.created_by,
        "template_used": c.template_used,
        "subtasks": [subtask_to_dict(s) for s in c.subtasks],
    }


def subtask_to_dict(s: Subtask) -> Dict[str, Any]:
    return {
        "id": s.id,
        "campaign_id": s.campaign_id,
        "name": s.name,
        "responsible": s.responsible,
        "budget_rub": s.budget_rub,
        "execution_days": s.execution_days,
        "deadline": s.deadline.date().isoformat() if s.deadline else None,
        "reminder_sent": bool(s.reminder_sent),
    }


def template_to_dict(t: CampaignTemplate) -> Dict[str, Any]:
    return {
        "id": t.id,
        "name": t.name,
        "description": t.description,
        "total_budget_rub": t.total_budget_rub,
        "participants_example": t.participants_example,
        "deadline_example": t.deadline_example,
        "subtasks": [
            {
                "name": st.name,
                "responsible_role": st.responsible_role,
                "budget_percentage": st.budget_percentage,
                "execution_days_ratio": st.execution_days_ratio,
            }
            for st in t.subtasks
        ],
    }


# --------------------------------------------------------------------------- #
# LLM-операции
# --------------------------------------------------------------------------- #
async def classify_message(text: str) -> str:
    return await llm_client.classify_message(text)


async def parse_campaign(text: str, user_display: Optional[str] = None) -> Dict[str, Any]:
    users = await list_users()
    parsed = await llm_client.parse_campaign(
        text, user_display=user_display, roster=roster_for_prompt(users)
    )
    parsed = resolve_self_references(parsed, user_display)
    parsed["participants"] = match_participants_to_roster(parsed.get("participants") or [], users)
    return parsed


# Слова, которыми пользователь называет самого себя в списке участников.
_SELF_TOKENS = {
    "я", "мне", "меня", "мной", "мою", "себя", "сам", "сама", "я сам", "я сама",
    "от меня", "ответственный я", "me", "myself", "i", "self",
}


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _display_user(name: str, role: Optional[str]) -> str:
    name = (name or "").strip()
    role = (role or "").strip()
    return f"{name} ({role})" if name and role else (name or role or "")


def _name_part(token: str) -> str:
    """«Иван (Дизайнер)» -> «Иван»; «Дизайнер» -> «Дизайнер»."""
    m = re.match(r"^(.*?)\s*\([^)]*\)\s*$", token or "")
    return (m.group(1) if m else (token or "")).strip()


def roster_for_prompt(users: List[Dict[str, Any]]) -> str:
    """Текстовый список известных участников чата для подсказки LLM."""
    if not users:
        return "(пока никто не представился)"
    return "\n".join(f"- {u['name']} — {u.get('role') or 'роль не указана'}" for u in users)


def match_participants_to_roster(
    participants: List[str], users: List[Dict[str, Any]]
) -> List[str]:
    """Сопоставляет упомянутых участников со списком людей в чате.

    - токен — имя известного человека → «Имя (Роль)» (или просто «Имя», если роль не задана);
    - токен — роль, и в чате есть люди с такой ролью → «Имя (Роль)» для каждого;
    - иначе токен остаётся как есть (с заглавной буквы).
    Дубликаты убираются, порядок сохраняется.
    """
    by_name: Dict[str, Dict[str, Any]] = {}
    by_role: Dict[str, List[Dict[str, Any]]] = {}
    for u in users:
        if _norm(u.get("name")):
            by_name[_norm(u["name"])] = u
        if _norm(u.get("role")):
            by_role.setdefault(_norm(u["role"]), []).append(u)

    out: List[str] = []

    def add(value: str) -> None:
        value = (value or "").strip()
        if value and value not in out:
            out.append(value)

    for raw in participants or []:
        token = (raw or "").strip()
        if not token:
            continue
        key = _norm(_name_part(token))
        if key in by_name:
            u = by_name[key]
            add(_display_user(u["name"], u.get("role")))
        elif key in by_role:
            for u in by_role[key]:
                add(_display_user(u["name"], u.get("role")))
        else:
            add(token[:1].upper() + token[1:])
    return out


def resolve_self_references(parsed: Dict[str, Any], user_display: Optional[str]) -> Dict[str, Any]:
    """Заменяет в participants само-упоминания пользователя («я», «дизайнер (я)») на его имя.

    ``user_display`` — строка вида «Имя (Роль)» или «Имя»; если её нет, ничего не меняем.
    """
    if not user_display:
        return parsed
    name_only = user_display.split(" (")[0].strip() or user_display
    out = dict(parsed)
    new_parts: List[str] = []
    for raw in out.get("participants") or []:
        token = (raw or "").strip()
        low = token.lower()
        if not token:
            continue
        if low in _SELF_TOKENS:
            repl = user_display
        else:
            m = re.match(r"^(.*?)\s*\(\s*([^)]+?)\s*\)\s*$", token)
            if m and m.group(2).strip().lower() in _SELF_TOKENS:
                role_part = m.group(1).strip()
                repl = f"{name_only} ({role_part})" if role_part else user_display
            else:
                repl = token
        if repl and repl not in new_parts:
            new_parts.append(repl)
    out["participants"] = new_parts
    return out


# --------------------------------------------------------------------------- #
# Участники чата (кто представился в веб-интерфейсе)
# --------------------------------------------------------------------------- #
def chat_user_to_dict(u: ChatUser) -> Dict[str, Any]:
    return {"id": u.id, "name": u.name, "role": u.role}


async def list_users() -> List[Dict[str, Any]]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(ChatUser).order_by(ChatUser.name))
        return [chat_user_to_dict(u) for u in result.scalars().all()]


async def register_user(name: str, role: Optional[str] = None) -> Dict[str, Any]:
    """Регистрирует/обновляет участника чата по имени. Пустое имя — ошибка."""
    name = (name or "").strip()
    role = (role or "").strip() or None
    if not name:
        raise CampaignError("Укажите имя.")
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(ChatUser).where(ChatUser.name == name))
        user = result.scalars().first()
        if user is None:
            user = ChatUser(name=name, role=role)
            session.add(user)
        elif role is not None:
            user.role = role
        await session.commit()
        await session.refresh(user)
        return chat_user_to_dict(user)


async def delete_user(name: str) -> bool:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(ChatUser).where(ChatUser.name == name))
        user = result.scalars().first()
        if user is None:
            return False
        await session.delete(user)
        await session.commit()
    return True


async def get_template_examples() -> str:
    """Возвращает текстовые примеры шаблонов для подсказки LLM."""
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(CampaignTemplate).options(selectinload(CampaignTemplate.subtasks))
            )
            templates = result.scalars().all()
            if not templates:
                return ""

            examples = []
            for template in templates[:2]:
                example = f"📌 Шаблон '{template.name}':\n"
                example += f"   Бюджет: {template.total_budget_rub}₽\n"
                example += f"   Участники: {template.participants_example}\n"
                example += f"   Дедлайн: {template.deadline_example}\n"
                example += "   Подзадачи:\n"
                for subtask in template.subtasks:
                    example += f"   - {subtask.name}\n"
                    example += f"     Ответственный: {subtask.responsible_role}\n"
                    example += f"     Бюджет: {subtask.budget_percentage}% от общего\n"
                    example += f"     Срок: {int(subtask.execution_days_ratio * 100)}% от дедлайна\n"
                examples.append(example)
            return "\n".join(examples)
    except Exception as e:
        logger.error(f"Failed to get template examples: {e}")
        return ""


async def answer_question(question: str) -> str:
    """Отвечает на вопрос пользователя, используя данные всех кампаний."""
    async with AsyncSessionLocal() as session:
        stmt = select(Campaign).options(selectinload(Campaign.subtasks))
        result = await session.execute(stmt)
        campaigns = result.scalars().all()

    if not campaigns:
        return "📭 Нет кампаний. Создайте кампанию для начала."

    campaigns_context = ""
    for campaign in campaigns:
        campaigns_context += f"\n📌 Кампания: {campaign.name}\n"
        campaigns_context += f"   Ответственные: {campaign.responsible}\n"
        campaigns_context += f"   Дедлайн: {campaign.deadline.date()}\n"
        campaigns_context += f"   Бюджет: {campaign.budget_rub} ₽\n"
        if campaign.subtasks:
            campaigns_context += "   Подзадачи:\n"
            for task in campaign.subtasks:
                campaigns_context += f"   - {task.name}\n"
                campaigns_context += f"     Ответственный: {task.responsible}\n"
                campaigns_context += f"     Срок: {task.execution_days} дней\n"
                campaigns_context += f"     Бюджет: {task.budget_rub} ₽\n"
        campaigns_context += "\n"

    return await llm_client.answer_question(question, campaigns_context)


# --------------------------------------------------------------------------- #
# Кампании
# --------------------------------------------------------------------------- #
async def list_campaigns() -> List[Dict[str, Any]]:
    async with AsyncSessionLocal() as session:
        stmt = select(Campaign).options(selectinload(Campaign.subtasks))
        result = await session.execute(stmt)
        campaigns = result.scalars().all()
    return [campaign_to_dict(c) for c in campaigns]


async def delete_campaign(name: str) -> bool:
    """Удаляет кампанию по названию. Возвращает True если удалена."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Campaign).where(Campaign.name == name))
        campaign = result.scalars().first()
        if not campaign:
            return False
        await session.delete(campaign)
        await session.commit()
    return True


class CampaignError(Exception):
    """Ошибка валидации/создания кампании, безопасная для показа пользователю."""


# Обязательные поля кампании и их человеко-читаемые названия (для уточняющих вопросов).
REQUIRED_CAMPAIGN_FIELDS = {
    "campaign_name": "название кампании",
    "deadline": "дедлайн (дату)",
    "participants": "участников",
    "budget_rub": "бюджет в рублях",
}


def _clean_str(value: Any) -> Optional[str]:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value or None


def enrich_parsed(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Достраивает поля, которые можно вывести без участия пользователя.

    Сейчас: если нет названия кампании, но есть задача — берём название из задачи
    (и наоборот). Возвращает новый словарь, исходный не меняет.
    """
    out = dict(parsed)
    name = _clean_str(out.get("campaign_name"))
    task = _clean_str(out.get("task"))
    if not name and task:
        name = task if len(task) <= 120 else task[:117].rstrip() + "…"
        out["campaign_name"] = name
    if not task and name:
        out["task"] = name
    return out


def missing_campaign_fields(parsed: Dict[str, Any]) -> List[str]:
    """Возвращает человеко-читаемые названия обязательных полей, которых не хватает."""
    missing: List[str] = []
    if not _clean_str(parsed.get("campaign_name")):
        missing.append(REQUIRED_CAMPAIGN_FIELDS["campaign_name"])
    if not _clean_str(parsed.get("deadline")):
        missing.append(REQUIRED_CAMPAIGN_FIELDS["deadline"])
    if not (parsed.get("participants") or []):
        missing.append(REQUIRED_CAMPAIGN_FIELDS["participants"])
    budget = parsed.get("budget_rub")
    if not budget or budget <= 0:
        missing.append(REQUIRED_CAMPAIGN_FIELDS["budget_rub"])
    return missing


def merge_parsed(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """Дозаполняет ``base`` данными из ``extra``: пустые поля берём из extra,
    список участников объединяем без дублей. Существующие непустые значения base
    имеют приоритет."""
    out = dict(base)
    for key in ("campaign_name", "task", "deadline"):
        if not _clean_str(out.get(key)) and _clean_str(extra.get(key)):
            out[key] = extra[key]
    if (not out.get("budget_rub") or out["budget_rub"] <= 0) and extra.get("budget_rub") and extra["budget_rub"] > 0:
        out["budget_rub"] = extra["budget_rub"]
    participants = list(out.get("participants") or [])
    for p in extra.get("participants") or []:
        if p not in participants:
            participants.append(p)
    out["participants"] = participants
    return out


# Слова, которые в списке участников означают роль/должность, а не имя человека.
# Если участник назван так — не подставляем ещё одну роль из шаблона рядом
# («дизайнер (Маркетолог)» — бессмыслица).
_ROLE_WORDS = {
    "маркетолог", "таргетолог", "дизайнер", "копирайтер", "контент-менеджер",
    "контент менеджер", "контентщик", "smm", "smm-менеджер", "смм", "смм-менеджер",
    "аналитик", "менеджер", "менеджер проекта", "проджект", "проджект-менеджер",
    "продакт", "продакт-менеджер", "разработчик", "программист", "верстальщик",
    "редактор", "корректор", "фотограф", "видеограф", "монтажёр", "монтажер",
    "продюсер", "пиарщик", "pr", "pr-менеджер", "директолог", "сеошник", "seo",
}
_ROLE_SUFFIXES = ("менеджер", "директор", "специалист", "-лид")


def _looks_like_role(token: str) -> bool:
    """Эвристика: ``token`` — это название роли/должности, а не имя конкретного человека."""
    t = (token or "").strip().lower()
    if not t:
        return False
    if t in _ROLE_WORDS:
        return True
    return any(t.endswith(suf) for suf in _ROLE_SUFFIXES)


def _format_responsible(person: str, role: str) -> str:
    """Формирует строку ответственного для подзадачи.

    Если ``person`` уже несёт роль (или сам является ролью) — не задваиваем,
    просто приводим к читаемому виду; иначе — «Имя (Роль)»."""
    person = (person or "").strip()
    if not person:
        return role
    if "(" in person or _looks_like_role(person):
        return person[:1].upper() + person[1:]
    return f"{person} ({role})"


def _subtasks_from_template(
    template: Dict[str, Any], participants: List[str], total_budget: float, days_to_deadline: int
) -> List[Dict[str, Any]]:
    """Детерминированно строит подзадачи из шаблона: бюджет = % от общего,
    срок = доля от дней до дедлайна, ответственные раздаются по кругу с указанием роли."""
    out: List[Dict[str, Any]] = []
    for i, st in enumerate(template.get("subtasks") or []):
        pct = float(st.get("budget_percentage") or 0)
        ratio = float(st.get("execution_days_ratio") or 0)
        budget = round(total_budget * pct / 100.0, 2)
        days = max(1, int(round(days_to_deadline * ratio))) if days_to_deadline > 0 else 1
        role = st.get("responsible_role") or "Ответственный"
        if participants:
            responsible = _format_responsible(participants[i % len(participants)], role)
        else:
            responsible = role
        out.append(
            {
                "name": st.get("name") or "Подзадача",
                "responsible": responsible,
                "execution_days": days,
                "budget_rub": budget,
            }
        )
    return out


def _template_distinct_roles(template: Dict[str, Any]) -> int:
    """Сколько разных ролей задействует шаблон."""
    roles = {
        (st.get("responsible_role") or "").strip().lower()
        for st in template.get("subtasks") or []
        if (st.get("responsible_role") or "").strip()
    }
    return len(roles)


def _team_covers_template(participants: List[str], template: Dict[str, Any]) -> bool:
    """True, если участников хватает на роли шаблона.

    Если людей меньше, чем разных ролей в шаблоне, навязывать шаблон не стоит —
    лучше сделать простое разбиение под реальную команду.
    """
    needed = _template_distinct_roles(template)
    return len(participants) >= max(1, needed)


async def _decompose(
    campaign_name: str,
    task: Optional[str],
    participants: List[str],
    today: datetime,
    deadline_dt: datetime,
    days_to_deadline: int,
    budget_rub: float,
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """Разбивает кампанию на подзадачи.

    LLM выбирает подходящий шаблон; если выбран И участников хватает на его роли —
    применяем формулы шаблона детерминированно. Иначе — свободное разбиение через LLM
    под реально указанную команду. Возвращает (список подзадач, имя применённого шаблона или None).
    """
    templates = await list_templates()
    chosen_name = await llm_client.pick_template(campaign_name, task, templates)
    if chosen_name:
        template = next((t for t in templates if t["name"] == chosen_name), None)
        if template and _team_covers_template(participants, template):
            subtasks = _subtasks_from_template(template, participants, budget_rub, days_to_deadline)
            if subtasks:
                return subtasks, chosen_name
        elif template:
            logger.info(
                "Шаблон '%s' подобран, но участников (%d) меньше, чем ролей (%d) — "
                "делаю свободное разбиение под команду.",
                chosen_name, len(participants), _template_distinct_roles(template),
            )

    template_examples = await get_template_examples()
    subtasks = await llm_client.decompose_task_llm(
        campaign_name,
        participants,
        today.date().isoformat(),
        deadline_dt.date().isoformat(),
        days_to_deadline,
        budget_rub,
        template_examples=template_examples,
    )
    return subtasks, None


async def create_campaign_from_parsed(
    parsed: Dict[str, Any], created_by: Optional[str] = None
) -> Dict[str, Any]:
    """Создаёт кампанию и подзадачи из распарсенных LLM данных.

    Бросает CampaignError с понятным текстом при невалидных данных.
    """
    parsed = enrich_parsed(parsed)
    campaign_name = _clean_str(parsed.get("campaign_name"))
    deadline_str = parsed.get("deadline")
    participants = parsed.get("participants") or []
    budget_rub = parsed.get("budget_rub") or 0
    responsible = ", ".join(participants) if participants else "Не указан"

    missing = missing_campaign_fields(parsed)
    if missing:
        raise CampaignError("Недостаточно данных для создания кампании: не хватает " + ", ".join(missing) + ".")

    deadline_dt = parse_date_to_datetime(deadline_str)
    if not deadline_dt:
        raise CampaignError(
            f"Неверный формат даты: '{deadline_str}'. Используйте ДД.ММ.ГГГГ или ГГГГ-ММ-ДД."
        )

    today = datetime.utcnow()
    days_to_deadline = (deadline_dt - today).days
    if days_to_deadline < 0:
        raise CampaignError("Дедлайн уже прошёл. Выберите будущую дату.")

    # Проверяем дубликат имени ДО дорогого вызова LLM (повторная проверка ниже —
    # на случай гонки между двумя запросами).
    async with AsyncSessionLocal() as session:
        existing = await session.execute(select(Campaign).where(Campaign.name == campaign_name))
        if existing.scalars().first():
            raise CampaignError(f"Кампания '{campaign_name}' уже существует.")

    try:
        subtasks, template_used = await _decompose(
            campaign_name, parsed.get("task"), participants, today, deadline_dt, days_to_deadline, budget_rub
        )
    except Exception as e:
        logger.error(f"Decompose error: {e}", exc_info=True)
        raise CampaignError(f"Не удалось разбить кампанию на задачи: {str(e)[:120]}")

    if not subtasks:
        template_used = None
        subtasks = [
            {
                "name": campaign_name,
                "responsible": participants[0] if participants else "admin",
                "execution_days": max(1, days_to_deadline),
                "budget_rub": budget_rub,
            }
        ]

    async with AsyncSessionLocal() as session:
        existing = await session.execute(select(Campaign).where(Campaign.name == campaign_name))
        if existing.scalars().first():
            raise CampaignError(f"Кампания '{campaign_name}' уже существует.")

        campaign = Campaign(
            name=campaign_name,
            responsible=responsible,
            deadline=deadline_dt,
            budget_rub=budget_rub,
            chat_id=None,
            created_by=created_by,
            template_used=template_used,
        )
        session.add(campaign)
        await session.flush()

        for st in subtasks:
            execution_days = max(1, int(round(float(st.get("execution_days", days_to_deadline)))))
            subtask_deadline = today + timedelta(days=execution_days)
            session.add(
                Subtask(
                    campaign_id=campaign.id,
                    name=st.get("name", "Подзадача"),
                    responsible=st.get("responsible", "Не указан"),
                    budget_rub=float(st.get("budget_rub", 0)),
                    execution_days=execution_days,
                    deadline=subtask_deadline,
                )
            )
        await session.commit()

        # Перечитываем со связями для ответа
        result = await session.execute(
            select(Campaign).where(Campaign.id == campaign.id).options(selectinload(Campaign.subtasks))
        )
        campaign = result.scalars().first()
        return campaign_to_dict(campaign)


# --------------------------------------------------------------------------- #
# Шаблоны
# --------------------------------------------------------------------------- #
async def list_templates() -> List[Dict[str, Any]]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(CampaignTemplate).options(selectinload(CampaignTemplate.subtasks))
        )
        templates = result.scalars().all()
    return [template_to_dict(t) for t in templates]


async def get_template(name: str) -> Optional[Dict[str, Any]]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(CampaignTemplate)
            .where(CampaignTemplate.name == name)
            .options(selectinload(CampaignTemplate.subtasks))
        )
        template = result.scalars().first()
    return template_to_dict(template) if template else None


def _validate_template_input(name: str, total_budget_rub: float, subtasks: List[Dict[str, Any]]) -> None:
    if not name or total_budget_rub <= 0 or not subtasks:
        raise CampaignError("Нужны название, положительный бюджет и хотя бы одна подзадача.")
    for st in subtasks:
        if not (st.get("name") or "").strip() or not (st.get("responsible_role") or "").strip():
            raise CampaignError("У каждой подзадачи нужны название и роль.")


async def _replace_template_subtasks(session, template_id: int, subtasks: List[Dict[str, Any]]) -> None:
    await session.execute(sa_delete(TemplateSubtask).where(TemplateSubtask.template_id == template_id))
    await session.flush()
    for st in subtasks:
        session.add(
            TemplateSubtask(
                template_id=template_id,
                name=str(st["name"]).strip(),
                responsible_role=str(st["responsible_role"]).strip(),
                budget_percentage=float(st["budget_percentage"]),
                execution_days_ratio=float(st["execution_days_ratio"]),
            )
        )


async def _reread_template(session, template_id: int) -> CampaignTemplate:
    result = await session.execute(
        select(CampaignTemplate)
        .where(CampaignTemplate.id == template_id)
        .options(selectinload(CampaignTemplate.subtasks))
    )
    return result.scalars().first()


async def add_template(
    name: str,
    description: str,
    total_budget_rub: float,
    participants_example: str,
    deadline_example: str,
    subtasks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Добавляет новый шаблон. Бросает CampaignError при дубликате/невалидных данных."""
    name = (name or "").strip()
    _validate_template_input(name, total_budget_rub, subtasks)

    async with AsyncSessionLocal() as session:
        existing = await session.execute(select(CampaignTemplate).where(CampaignTemplate.name == name))
        if existing.scalars().first():
            raise CampaignError(f"Шаблон '{name}' уже существует.")

        template = CampaignTemplate(
            name=name,
            description=description,
            total_budget_rub=total_budget_rub,
            participants_example=participants_example,
            deadline_example=deadline_example,
            created_by=None,
        )
        session.add(template)
        await session.flush()
        await _replace_template_subtasks(session, template.id, subtasks)
        await session.commit()
        return template_to_dict(await _reread_template(session, template.id))


async def update_template(
    name: str,
    description: str,
    total_budget_rub: float,
    participants_example: str,
    deadline_example: str,
    subtasks: List[Dict[str, Any]],
    new_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Перезаписывает существующий шаблон (метаданные + список подзадач целиком).

    Если ``new_name`` задан и отличается — переименовывает шаблон, проверив, что
    имя свободно. Бросает CampaignError, если шаблон не найден или данные невалидны.
    """
    target_name = ((new_name if new_name is not None else name) or "").strip()
    _validate_template_input(target_name, total_budget_rub, subtasks)

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(CampaignTemplate).where(CampaignTemplate.name == name))
        template = result.scalars().first()
        if template is None:
            raise CampaignError(f"Шаблон '{name}' не найден.")

        if target_name != template.name:
            clash = await session.execute(
                select(CampaignTemplate).where(CampaignTemplate.name == target_name)
            )
            if clash.scalars().first():
                raise CampaignError(f"Шаблон '{target_name}' уже существует.")

        template.name = target_name
        template.description = description
        template.total_budget_rub = total_budget_rub
        template.participants_example = participants_example
        template.deadline_example = deadline_example
        await session.flush()
        await _replace_template_subtasks(session, template.id, subtasks)
        await session.commit()
        return template_to_dict(await _reread_template(session, template.id))


async def delete_template(name: str) -> bool:
    """Удаляет шаблон по названию. Возвращает True, если удалён."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(CampaignTemplate).where(CampaignTemplate.name == name))
        template = result.scalars().first()
        if template is None:
            return False
        await session.delete(template)
        await session.commit()
    return True


# --------------------------------------------------------------------------- #
# Ожидающие подтверждения кампании (хранятся в БД)
# --------------------------------------------------------------------------- #
async def create_pending(parsed: Dict[str, Any]) -> str:
    """Сохраняет распарсенную кампанию и возвращает её id для подтверждения."""
    pending_id = str(uuid.uuid4())
    async with AsyncSessionLocal() as session:
        session.add(
            PendingCampaign(id=pending_id, data=json.dumps(parsed), created_at=datetime.utcnow())
        )
        await session.commit()
    return pending_id


async def update_pending(pending_id: str, parsed: Dict[str, Any]) -> bool:
    """Перезаписывает данные ожидающей кампании. Возвращает False, если её нет."""
    async with AsyncSessionLocal() as session:
        row = await session.get(PendingCampaign, pending_id)
        if row is None:
            return False
        row.data = json.dumps(parsed)
        await session.commit()
        return True


async def get_pending(pending_id: str) -> Optional[Dict[str, Any]]:
    async with AsyncSessionLocal() as session:
        row = await session.get(PendingCampaign, pending_id)
        if row is None:
            return None
        return json.loads(row.data)


async def delete_pending(pending_id: str) -> None:
    async with AsyncSessionLocal() as session:
        await session.execute(sa_delete(PendingCampaign).where(PendingCampaign.id == pending_id))
        await session.commit()


async def cleanup_pending() -> int:
    """Удаляет просроченные ожидающие кампании. Возвращает число удалённых."""
    cutoff = datetime.utcnow() - timedelta(hours=PENDING_TTL_HOURS)
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            sa_delete(PendingCampaign).where(PendingCampaign.created_at < cutoff)
        )
        await session.commit()
        return result.rowcount or 0


# --------------------------------------------------------------------------- #
# Напоминания: подзадачи с дедлайном сегодня
# --------------------------------------------------------------------------- #
async def subtasks_due_today() -> List[Dict[str, Any]]:
    """Возвращает подзадачи, дедлайн которых приходится на сегодня (UTC)."""
    async with AsyncSessionLocal() as session:
        today = datetime.utcnow().date()
        stmt = (
            select(Subtask)
            .where(
                and_(
                    Subtask.deadline >= datetime.combine(today, datetime.min.time()),
                    Subtask.deadline < datetime.combine(today + timedelta(days=1), datetime.min.time()),
                )
            )
            .options(selectinload(Subtask.campaign))
        )
        result = await session.execute(stmt)
        subtasks = result.scalars().all()

        items = []
        for s in subtasks:
            d = subtask_to_dict(s)
            d["campaign_name"] = s.campaign.name if s.campaign else None
            items.append(d)
        return items
