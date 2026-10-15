FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY src ./src
COPY app ./app
COPY models ./models

EXPOSE 8000
CMD ["gunicorn", "app.app:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "60"]
