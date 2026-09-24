FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY recsys/ recsys/
COPY artifacts/ artifacts/
ENV OMP_NUM_THREADS=1 PORT=8000
EXPOSE 8000
CMD uvicorn recsys.app:app --host 0.0.0.0 --port $PORT --workers ${WORKERS:-4}
