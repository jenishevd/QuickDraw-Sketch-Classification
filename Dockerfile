FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN grep -vE '^(torch|torchvision)$' requirements.txt > requirements-rest.txt \
    && pip install --no-cache-dir -r requirements-rest.txt

COPY . .

EXPOSE 7860

CMD ["python", "demo.py"]
