FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        cron \
        tini \
        ffmpeg \
        ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY batinapapka.py ./
COPY entrypoint.sh /usr/local/bin/batinapapka-entrypoint
RUN chmod +x /usr/local/bin/batinapapka-entrypoint \
 && printf '#!/bin/sh\nexec /usr/local/bin/python /app/batinapapka.py "$@"\n' > /usr/local/bin/batinapapka \
 && chmod +x /usr/local/bin/batinapapka

VOLUME ["/state", "/videos"]
WORKDIR /state

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/batinapapka-entrypoint"]
CMD ["--recursive", "/videos"]
