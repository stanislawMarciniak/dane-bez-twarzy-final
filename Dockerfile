FROM python:3.10-slim


ENV PYTHONUNBUFFERED=1
ENV LANG C.UTF-8
ENV LANGUAGE C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR /app


COPY requirements.txt .


RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*


COPY . .


ENTRYPOINT ["python", "pipeline.py"]


CMD ["data/orig.txt"]