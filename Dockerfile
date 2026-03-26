FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    blender \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --default-timeout=600 --retries 3 \
    numpy scipy trimesh tqdm

RUN pip install --no-cache-dir --default-timeout=600 --retries 3 \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY flame_repair/ /app/flame_repair/
COPY run.py /app/run.py

RUN mkdir -p /app/models /app/input /app/output

ENTRYPOINT ["python", "run.py"]
CMD ["--help"]
