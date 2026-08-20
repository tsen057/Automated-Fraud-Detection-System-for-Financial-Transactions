FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run the training/evaluation pipeline.
# Override with `docker run <image> streamlit run dashboard/app.py --server.address 0.0.0.0`
# to serve the dashboard instead.
CMD ["python", "main.py"]
