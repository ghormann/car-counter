FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLOv8n weights into the image
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY src/ ./src/

EXPOSE 9600

ENTRYPOINT ["python", "-m", "src"]
