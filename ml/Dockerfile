FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --upgrade pip \
    && pip install fastapi uvicorn pandas numpy scikit-learn sqlalchemy psycopg2-binary joblib mlflow matplotlib seaborn

EXPOSE 8000 5000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
