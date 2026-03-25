# =============================================================================
# CPU image + noVNC — macOS Docker Desktop dev (build target: `cpu`)
# =============================================================================
# Ubuntu 22.04: system python3 is 3.10, and python3-tk matches that interpreter.
# The official python:3.10-slim-bookworm image installs Python under /usr/local; apt's
# python3-tk on Bookworm targets Debian's python3 (3.11), so Tkinter would be missing at runtime.
FROM ubuntu:22.04 AS cpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-tk \
    tk \
    libgl1 \
    libglib2.0-0 \
    xvfb \
    x11vnc \
    novnc \
    fluxbox \
    x11-xserver-utils \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python3

WORKDIR /app

COPY requirements-docker-cpu.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir -r requirements-docker-cpu.txt

COPY . .
COPY docker/entrypoint-novnc.sh /entrypoint-novnc.sh
RUN chmod +x /entrypoint-novnc.sh

EXPOSE 6080

ENTRYPOINT ["/entrypoint-novnc.sh"]

# =============================================================================
# GPU image — Linux + NVIDIA Container Toolkit (default build stage: `gpu`)
# =============================================================================
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 AS gpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-tk \
    tk \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel \
    && pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "ui/main_ui.py"]
