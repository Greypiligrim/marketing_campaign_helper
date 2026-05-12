from datetime import datetime
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from config import settings

Base = declarative_base()


class Campaign(Base):
    __tablename__ = "campaigns"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    responsible = Column(String(500), nullable=False)  # Ответственные
    deadline = Column(DateTime, nullable=False)
    budget_rub = Column(Float, nullable=False)
    chat_id = Column(Integer, nullable=True)
    created_by = Column(String(255), nullable=True)  # Кто создал: "Имя (Роль)"
    template_used = Column(String(255), nullable=True)  # Имя применённого шаблона, если был

    subtasks = relationship("Subtask", back_populates="campaign", cascade="all, delete-orphan")


class Subtask(Base):
    __tablename__ = "subtasks"

    id = Column(Integer, primary_key=True)
    campaign_id = Column(Integer, ForeignKey("campaigns.id"), nullable=False)
    name = Column(String(255), nullable=False)
    responsible = Column(String(255), nullable=False)
    budget_rub = Column(Float, nullable=False)
    execution_days = Column(Integer, nullable=False)  # Срок выполнения в днях (целое число)
    deadline = Column(DateTime, nullable=False)  # Дата дедлайна подзадачи
    reminder_sent = Column(Integer, default=0)  # 0 = не отправлено, 1 = отправлено

    campaign = relationship("Campaign", back_populates="subtasks")


class CampaignTemplate(Base):
    __tablename__ = "campaign_templates"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)  # Название шаблона
    description = Column(Text, nullable=True)  # Описание
    total_budget_rub = Column(Float, nullable=False)  # Примерный бюджет
    participants_example = Column(String(500), nullable=False)  # Пример участников
    deadline_example = Column(String(100), nullable=False)  # Пример дедлайна

    subtasks = relationship("TemplateSubtask", back_populates="template", cascade="all, delete-orphan")
    created_by = Column(Integer, nullable=True)  # Telegram User ID кто создал


class TemplateSubtask(Base):
    __tablename__ = "template_subtasks"

    id = Column(Integer, primary_key=True)
    template_id = Column(Integer, ForeignKey("campaign_templates.id"), nullable=False)
    name = Column(String(255), nullable=False)
    responsible_role = Column(String(255), nullable=False)  # Роль (например: "Маркетолог")
    budget_percentage = Column(Float, nullable=False)  # Процент от бюджета (0-100)
    execution_days_ratio = Column(Float, nullable=False)  # Отношение к дням дедлайна (0-1)

    template = relationship("CampaignTemplate", back_populates="subtasks")


class ChatUser(Base):
    """Участник чата: тот, кто представился в веб-интерфейсе (имя + зона ответственности).

    Используется, чтобы LLM при разборе кампании понимал, «кто есть кто», и подставлял
    «Имя (Роль)» вместо голых имён/ролей. Список показывается в боковом меню.
    """
    __tablename__ = "chat_users"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    role = Column(String(255), nullable=True)  # «Дизайнер», «Копирайтер» и т.п.
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class PendingCampaign(Base):
    """Распарсенная LLM, но ещё не подтверждённая кампания.

    Хранится в БД (а не в памяти процесса), чтобы переживать перезапуск
    сервиса и работать при нескольких воркерах.
    """
    __tablename__ = "pending_campaigns"

    id = Column(String(36), primary_key=True)  # uuid4
    data = Column(Text, nullable=False)  # JSON с распарсенными полями кампании
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


async_engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
)

AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=__import__("sqlalchemy.ext.asyncio").ext.asyncio.AsyncSession,
    expire_on_commit=False,
)


def _migrate_sqlite(conn) -> None:
    """Мини-миграции для SQLite: добавляет недостающие колонки в существующие таблицы.

    create_all создаёт только отсутствующие таблицы, но не меняет уже существующие,
    поэтому новые колонки на старой БД приходится добавлять вручную.
    Выполняется внутри run_sync, поэтому conn — синхронное соединение.
    """
    def _columns(table: str):
        rows = conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
        return {r[1] for r in rows}

    campaigns_cols = _columns("campaigns")
    if "created_by" not in campaigns_cols:
        conn.exec_driver_sql("ALTER TABLE campaigns ADD COLUMN created_by VARCHAR(255)")
    if "template_used" not in campaigns_cols:
        conn.exec_driver_sql("ALTER TABLE campaigns ADD COLUMN template_used VARCHAR(255)")


async def create_tables() -> None:
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_migrate_sqlite)


async def init_default_templates() -> None:
    """Инициализирует встроенные шаблоны кампаний в БД"""
    from config.templates import BUILT_IN_TEMPLATES
    
    async with AsyncSessionLocal() as session:
        for template_name, template_data in BUILT_IN_TEMPLATES.items():
            # Проверяем, существует ли уже такой шаблон
            result = await session.execute(
                __import__("sqlalchemy").select(CampaignTemplate).where(
                    CampaignTemplate.name == template_name
                )
            )
            existing = result.scalars().first()
            
            if existing:
                continue  # Шаблон уже существует, пропускаем
            
            # Создаём новый шаблон
            template = CampaignTemplate(
                name=template_name,
                description=template_data.get("description"),
                total_budget_rub=template_data.get("total_budget", 0),
                participants_example=template_data.get("example_participants", ""),
                deadline_example=template_data.get("example_deadline", ""),
            )
            session.add(template)
            await session.flush()
            
            # Добавляем подзадачи к шаблону
            for subtask_data in template_data.get("subtasks", []):
                subtask = TemplateSubtask(
                    template_id=template.id,
                    name=subtask_data.get("name", ""),
                    responsible_role=subtask_data.get("responsible_role", ""),
                    budget_percentage=subtask_data.get("budget_percentage", 0),
                    execution_days_ratio=subtask_data.get("execution_days_ratio", 0),
                )
                session.add(subtask)
        
        await session.commit()