FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HOST=0.0.0.0 \
    PORT=8000 \
    DEBUG=False

EXPOSE 8000

# OPENROUTER_API_KEY передаётся через -e / окружение при запуске.
CMD ["python", "main.py"]
