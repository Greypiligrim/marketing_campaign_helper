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


async def create_tables() -> None:
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)