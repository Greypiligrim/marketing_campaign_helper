import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "openai/gpt-5-nano"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
DATABASE_URL = "sqlite+aiosqlite:///./marketing_bot.db"
NOTIFICATION_TIME = "09:00"  # UTC
DEBUG = os.getenv("DEBUG", "True") == "True"
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
