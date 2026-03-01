FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY api.py /app/api.py
COPY namonexus_fusion /app/namonexus_fusion
COPY README.md NOTICE.txt /app/

RUN groupadd --system --gid 10001 namonexus \
    && useradd --system --uid 10001 --gid 10001 --create-home --home-dir /home/namonexus --shell /usr/sbin/nologin namonexus \
    && chown -R namonexus:namonexus /app

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/health', timeout=3)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
