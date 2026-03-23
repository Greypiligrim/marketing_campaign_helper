import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from dateparser import parse as parse_date
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest
from telegram.error import TimedOut

from config import settings
from db.models import AsyncSessionLocal, Campaign, Subtask, create_tables
from ml.llm import NemotronLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm_client = NemotronLLM()


def format_campaign_preview(c: Dict[str, Any]) -> str:
    """Форматирует preview кампании для показа пользователю"""
    participants = ', '.join(c.get('participants') or [])
    deadline = c.get('deadline')
    
    # Форматируем дату
    if deadline and isinstance(deadline, str):
        try:
            deadline = datetime.fromisoformat(deadline).strftime('%d.%m.%Y')
        except (ValueError, TypeError):
            deadline = str(deadline) if deadline else "Не указан"
    else:
        deadline = "Не указан"
    
    return (
        f"📣 Название: {c.get('campaign_name')}\n"
        f"🎯 Задача: {c.get('task')}\n"
        f"⏰ Дедлайн: {deadline}\n"
        f"👥 Участники: {participants}\n"
        f"💰 Бюджет: {c.get('budget_rub')} ₽\n"
        f"✅ Confidence: {c.get('confidence_score')}"
    )


def parse_date_to_datetime(date_str: Optional[str]) -> Optional[datetime]:
    """Парсит дату из строки в datetime"""
    if not date_str:
        return None
    
    # Если это уже ISO формат (YYYY-MM-DD), парсим напрямую
    if isinstance(date_str, str) and len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
        try:
            return datetime.fromisoformat(date_str)
        except ValueError:
            pass
    
    # Иначе используем dateparser для русских дат
    dt = parse_date(date_str, languages=['ru', 'en'], settings={'DATE_ORDER': 'DMY'})
    return dt if dt else None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start"""
    await update.message.reply_text(
        "👋 Привет! Я маркетинговый бот.\n\n"
        "Команды:\n"
        "📨 Отправьте описание кампании\n"
        "❓ Задайте вопрос о кампаниях\n"
        "/list - Список всех кампаний\n"
        "/delete <название> - Удалить кампанию"
    )


async def list_campaigns(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /list - показывает все кампании"""
    chat_id = update.effective_chat.id
    async with AsyncSessionLocal() as session:
        stmt = select(Campaign).where(Campaign.chat_id == chat_id).options(selectinload(Campaign.subtasks))
        result = await session.execute(stmt)
        campaigns = result.scalars().all()
    
    if not campaigns:
        await update.message.reply_text("📭 Нет кампаний.")
        return
    
    text = "📋 Ваши кампании:\n\n"
    for c in campaigns:
        text += f"• {c.name}\n"
        text += f"  Ответственные: {c.responsible}\n"
        text += f"  Дедлайн: {c.deadline.date()}\n"
        text += f"  Бюджет: {c.budget_rub}₽\n"
        text += f"  Подзадач: {len(c.subtasks)}\n\n"
    
    await update.message.reply_text(text)


async def delete_campaign(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /delete"""
    if not context.args:
        await update.message.reply_text("❌ Укажите название: /delete <название>")
        return
    
    name = " ".join(context.args)
    chat_id = update.effective_chat.id
    
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Campaign).where(Campaign.chat_id == chat_id, Campaign.name == name)
        )
        campaign = result.scalars().first()
        if not campaign:
            await update.message.reply_text(f"❌ Кампания '{name}' не найдена.")
            return
        
        await session.delete(campaign)
        await session.commit()
    
    await update.message.reply_text(f"✅ Кампания '{name}' удалена.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик обычных сообщений"""
    user_text = update.message.text
    if not user_text:
        return

    try:
        msg_type = await llm_client.classify_message(user_text)
        logger.info(f"Message classified as: {msg_type}")
    except Exception as e:
        logger.error(f"Classify error: {e}", exc_info=True)
        await update.message.reply_text("❌ Ошибка обработки сообщения.")
        return

    if msg_type == "question":
        # Обработка вопроса о кампаниях
        try:
            chat_id = update.effective_chat.id
            async with AsyncSessionLocal() as session:
                # Получаем все кампании с их подзадачами (явно загружаем через selectinload)
                stmt = select(Campaign).where(Campaign.chat_id == chat_id).options(selectinload(Campaign.subtasks))
                result = await session.execute(stmt)
                campaigns = result.scalars().all()
                
                if not campaigns:
                    await update.message.reply_text("📭 Нет кампаний. Создайте кампанию для начала.")
                    return
                
                # Форматируем контекст с кампаниями и подзадачами
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
                
                answer = await llm_client.answer_question(user_text, campaigns_context)
            
            await update.message.reply_text(answer)
        except Exception as e:
            logger.error(f"Answer error: {e}", exc_info=True)
            await update.message.reply_text("❌ Не удалось ответить на вопрос.")
    
    elif msg_type == "other":
        await update.message.reply_text(
            "👋 Я бот для управления маркетинговыми кампаниями.\n\n"
            "Отправьте описание кампании (с названием, задачей, бюджетом, дедлайном)\n"
            "или задайте вопрос о существующих кампаниях."
        )
    
    else:  # campaign
        try:
            parsed = await llm_client.parse_campaign(user_text)
            logger.info(f"Parsed campaign: {parsed}")
            
            # Сохраняем данные с уникальным ID
            campaign_id = str(uuid.uuid4())
            context.chat_data[campaign_id] = parsed
            
            # Показываем preview и кнопки подтверждения
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ Создать", callback_data=f"confirm:{campaign_id}"),
                 InlineKeyboardButton("❌ Отмена", callback_data="cancel")]
            ])
            try:
                await update.message.reply_text(format_campaign_preview(parsed), reply_markup=keyboard)
            except TimedOut:
                logger.warning("Telegram API timeout, retrying...")
                await asyncio.sleep(2)
                await update.message.reply_text(format_campaign_preview(parsed), reply_markup=keyboard)
        except TimedOut as e:
            logger.error(f"Timeout error: {e}")
            await update.message.reply_text("⏱️ Timeout при обработке. Попробуйте позже.")
        except Exception as e:
            logger.error(f"Parse campaign error: {e}", exc_info=True)
            await update.message.reply_text("❌ Не удалось распознать кампанию. Проверьте формат.")


async def confirm_create(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик нажатия кнопки 'Создать'"""
    query = update.callback_query
    if not query or not query.data:
        return
    
    try:
        # Ответ на callback query СРАЗУ, но без уведомления
        await query.answer()
    except Exception as e:
        logger.warning(f"Failed to answer callback query: {e}")

    campaign_id = query.data.replace("confirm:", "", 1)
    data = context.chat_data.pop(campaign_id, None)

    if not data:
        await query.edit_message_text("❌ Ошибка: данные кампании не найдены или устарели.")
        return

    # Извлекаем данные
    campaign_name = data.get('campaign_name')
    deadline_str = data.get('deadline')
    participants = data.get('participants') or []
    budget_rub = data.get('budget_rub') or 0
    responsible = ', '.join(participants) if participants else 'Не указан'

    # Базовая валидация
    if not campaign_name or not deadline_str or not responsible or budget_rub <= 0:
        await query.edit_message_text("❌ Недостаточно данных для создания кампании.")
        return

    # Парсим дату
    logger.info(f"Parsing campaign deadline: '{deadline_str}'")
    deadline_dt = parse_date_to_datetime(deadline_str)
    if not deadline_dt:
        logger.warning(f"Failed to parse campaign deadline: '{deadline_str}'")
        await query.edit_message_text(
            f"❌ Неверный формат даты: '{deadline_str}'.\n"
            f"Используйте формат ДД.ММ.ГГГГ или YYYY-MM-DD"
        )
        context.chat_data[campaign_id] = data
        return

    # Считаем дни до дедлайна
    today = datetime.utcnow()
    days_to_deadline = (deadline_dt - today).days
    
    if days_to_deadline <= 0:
        await query.edit_message_text("❌ Дедлайн уже прошел или наступает сегодня. Выберите будущую дату.")
        return

    # Сообщаем что начали обработку
    await query.edit_message_text("⏳ Разбиваю кампанию на задачи...")

    # Пытаемся разбить на подзадачи
    try:
        subtasks = await llm_client.decompose_task_llm(
            campaign_name, 
            participants, 
            today.date().isoformat(),
            deadline_dt.date().isoformat(), 
            days_to_deadline,
            budget_rub
        )
        logger.info(f"Decomposed into {len(subtasks)} subtasks")
    except Exception as e:
        logger.error(f"Decompose error: {e}", exc_info=True)
        await query.edit_message_text(
            f"❌ Не удалось разбить кампанию на задачи.\n"
            f"Ошибка: {str(e)[:100]}"
        )
        return

    # Если подзадач не получилось, создаём одну по умолчанию
    if not subtasks:
        logger.warning("No subtasks returned, creating default")
        subtasks = [{
            "name": campaign_name,
            "responsible": participants[0] if participants else "admin",
            "execution_days": days_to_deadline,
            "budget_rub": budget_rub
        }]

    # Сохраняем в БД
    try:
        async with AsyncSessionLocal() as session:
            # Создаём кампанию
            campaign = Campaign(
                name=campaign_name,
                responsible=responsible,
                deadline=deadline_dt,
                budget_rub=budget_rub,
                chat_id=update.effective_chat.id,
            )
            session.add(campaign)
            await session.flush()
            
            # Создаём подзадачи
            for st in subtasks:
                # Округляем до целого числа, минимум 1 день
                execution_days = max(1, int(round(float(st.get('execution_days', days_to_deadline)))))
                # Рассчитываем дедлайн подзадачи как сегодня + execution_days
                subtask_deadline = today + timedelta(days=execution_days)
                
                subtask = Subtask(
                    campaign_id=campaign.id,
                    name=st.get('name', 'Подзадача'),
                    responsible=st.get('responsible', 'Не указан'),
                    budget_rub=float(st.get('budget_rub', 0)),
                    execution_days=execution_days,
                    deadline=subtask_deadline,
                )
                session.add(subtask)
            
            await session.commit()
        
        # Успех!
        await query.edit_message_text(
            f"✅ Кампания '{campaign_name}' создана!\n"
            f"📋 Создано {len(subtasks)} подзадач."
        )
        logger.info(f"Campaign '{campaign_name}' created with {len(subtasks)} subtasks")
    
    except Exception as e:
        logger.error(f"Database error: {e}", exc_info=True)
        await query.edit_message_text(f"❌ Ошибка сохранения в БД: {str(e)[:100]}")


async def cancel_create(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик нажатия кнопки 'Отмена'"""
    query = update.callback_query
    if query:
        try:
            await query.answer()
            await query.edit_message_text("❌ Создание отменено.")
        except Exception as e:
            logger.warning(f"Cancel callback error: {e}")


async def send_reminders(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет напоминания о подзадачах в день дедлайна"""
    try:
        async with AsyncSessionLocal() as session:
            # Получаем подзадачи которые должны быть выполнены сегодня и напоминание ещё не отправлено
            from sqlalchemy import and_
            today = datetime.utcnow().date()
            
            stmt = select(Subtask).where(
                and_(
                    Subtask.reminder_sent == 0,
                    Subtask.deadline >= datetime.combine(today, datetime.min.time()),
                    Subtask.deadline < datetime.combine(today + timedelta(days=1), datetime.min.time())
                )
            ).options(selectinload(Subtask.campaign))
            
            result = await session.execute(stmt)
            subtasks = result.scalars().all()
            
            for subtask in subtasks:
                chat_id = subtask.campaign.chat_id
                if chat_id:
                    msg = (
                        f"⏰ НАПОМИНАНИЕ О ПОДЗАДАЧЕ!\n\n"
                        f"📌 Кампания: {subtask.campaign.name}\n"
                        f"📋 Подзадача: {subtask.name}\n"
                        f"👤 Ответственный: {subtask.responsible}\n"
                        f"💰 Бюджет: {subtask.budget_rub} ₽\n"
                        f"🎯 Дедлайн: СЕГОДНЯ!\n"
                    )
                    try:
                        await context.bot.send_message(chat_id=chat_id, text=msg)
                        # Отмечаем что напоминание отправлено
                        subtask.reminder_sent = 1
                        session.add(subtask)
                        await session.commit()
                        logger.info(f"Reminder sent for subtask {subtask.id}")
                    except Exception as e:
                        logger.error(f"Failed to send reminder for subtask {subtask.id}: {e}")
    except Exception as e:
        logger.error(f"send_reminders error: {e}", exc_info=True)


def setup_bot(app: Application) -> None:
    """Регистрируем все обработчики и job_queue для напоминаний"""
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('list', list_campaigns))
    app.add_handler(CommandHandler('delete', delete_campaign))
    app.add_handler(CallbackQueryHandler(confirm_create, pattern='^confirm:'))
    app.add_handler(CallbackQueryHandler(cancel_create, pattern='^cancel$'))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Добавляем job для проверки напоминаний каждый час
    app.job_queue.run_repeating(send_reminders, interval=3600, first=10)
    logger.info("Job queue для напоминаний добавлена")


def main() -> None:
    """Главная функция"""
    # Инициализируем БД
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(create_tables())

    # Создаём и запускаем бота
    request = HTTPXRequest(connect_timeout=settings.TELEGRAM_TIMEOUT, read_timeout=settings.TELEGRAM_TIMEOUT)
    app = Application.builder().token(settings.TELEGRAM_TOKEN).request(request).build()
    setup_bot(app)
    
    logger.info("🤖 Бот запущен...")
    app.run_polling()


if __name__ == "__main__":
    main()