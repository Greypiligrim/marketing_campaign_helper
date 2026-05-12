"""Точка входа: запускает веб-сервис (FastAPI + uvicorn).

Старый Telegram-бот сохранён в main_telegram_legacy.py для справки
(он требует пакет python-telegram-bot, которого больше нет в requirements.txt).
"""
import uvicorn

from config import settings


def main() -> None:
    uvicorn.run(
        "web.app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )


if __name__ == "__main__":
    main()
