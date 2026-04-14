FROM ubuntu:22.04 AS cpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/opt/ros/humble/lib/python3.10/dist-packages

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg2 lsb-release software-properties-common \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
       -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
       http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
       > /etc/apt/sources.list.d/ros2.list \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    xvfb \
    x11vnc \
    novnc \
    fluxbox \
    x11-xserver-utils \
    ros-humble-rosbag2 \
    ros-humble-sensor-msgs \
    ros-humble-cv-bridge \
    python3-rosbag2 \
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

FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 AS gpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/opt/ros/humble/lib/python3.10/dist-packages

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg2 lsb-release software-properties-common \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
       -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
       http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
       > /etc/apt/sources.list.d/ros2.list \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    ros-humble-rosbag2 \
    ros-humble-sensor-msgs \
    ros-humble-cv-bridge \
    python3-rosbag2 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel \
    && pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "-m", "on_device_app"]
