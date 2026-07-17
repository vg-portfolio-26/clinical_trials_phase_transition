FROM python:3.11-slim

WORKDIR /app

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libpangoft2-1.0-0 \
    libcairo2 \
    libglib2.0-0 \
    libgdk-pixbuf-xlib-2.0-0 \
    libffi-dev \
    libxml2 \
    libxslt1.1 \
    libfreetype6 \
    fontconfig \
    fonts-dejavu-core \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir \
    pandas==2.2.3 \
    datasets==4.8.5 \
    huggingface_hub==1.14.0 \
    scipy==1.17.1 \
    rapidfuzz==3.14.5 \
    matplotlib==3.8.0 \
    seaborn==0.13.2 \
    markdown==3.5.2 \
    weasyprint==60.0 \
    pydyf==0.10.0 \
    lifelines==0.30.3

# Default command (override with docker-compose as needed)
CMD ["python", "scripts/run_pipeline.py"]