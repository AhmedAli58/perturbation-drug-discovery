FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PDD_CONFIG_PATH=/app/configs/base.yaml

COPY pyproject.toml README.md ./
COPY configs ./configs
COPY docs ./docs
COPY src ./src

RUN pip install --upgrade pip \
    && pip install .

EXPOSE 8000

CMD ["uvicorn", "perturbation_dd.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
