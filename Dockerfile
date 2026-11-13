# Backend (CPU by default). GPU: build FROM a CUDA-enabled PyTorch image or install the cu* wheels matching
# your driver (see requirements.txt note) and run with `--gpus all`. Not built or run in the authoring sandbox.
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir -e . --no-deps
# model artifacts + palette are mounted, never baked in:  -v ./artifacts:/app/artifacts
ENV BRAINSENTINEL_MODEL_DIR=""
EXPOSE 8000
CMD ["uvicorn", "brainsentinel.api.main:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
