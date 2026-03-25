# OpenRoboVision

OpenRoboVision is an open-source framework for developing and deploying deep learning models tailored for robotics tasks. The repository includes scripts for training and evaluating object detection models, along with implementations for open-world object detection and segmentation. A user interface has been developed to facilitate human interactive learning, supporting active and continual learning approaches.

## Installation

### Using Conda

The following procedure has been tested on Ubuntu 22.04 LTS, with cuda 12.5.

1. Create environment
   ```bash 
   conda create -n orv_env python=3.10

2. Close the repository
   
3. Install pre-requisites
   ```bash
   cd open_robo_vision
   pip install -r requirements.txt
   ```

   **Note:**
   If you had an error about ```bash ModuleNotFoundError: No module named 'six'```, use this installation command: ```bash pip install --ignore-installed six```.

   **Note:**
   If you had an error about ```bash ModuleNotFoundError: No module named 'yaml'```, use this installation command: ```bash pip install --ignore-installed pyyaml```.

   **Note:**
   If you had an error about ```bash ModuleNotFoundError: No module named 'msgpack'```, use this installation command: ```bash pip install --ignore-installed msgpack```.

   **Note:**
   If you had an error about ```bash ModuleNotFoundError: No module named 'rich'```, use this installation command: ```bash pip install --ignore-installed rich```.

   **Note:**
   If you had an error about ```bash ModuleNotFoundError: No module named 'google'```, use this installation command: ```bash pip install --ignore-installed google```.

   **Note:**
   If you had an error about ```bash ModuleNotFoundError: No module named 'protobuf'```, use this installation command: ```bash pip install --ignore-installed protobuf```.

   **Note:**
   If you had an error about ```bash ModuleNotFoundError: No module named 'wrapt'```, use this installation command: ```bash pip install --ignore-installed wrapt```.

   **Note:**
   If you had an error about ```bash ModuleNotFoundError: No module named 'requests'```, use this installation command: ```bash pip install --ignore-installed requests```.

   **Note:**
   If you had an error about ```bash ModuleNotFoundError: No module named 'matplotlib'```, use this installation command: ```bash pip install --ignore-installed matplotlib```.

### macOS (Apple Silicon or Intel)

The instructions above target Ubuntu. On macOS:

1. **Install Tk for Python** (required for the UI; Homebrew Python does not include it):
   ```bash
   brew install python-tk@3.12
   ```

2. **Use a virtual environment** (recommended; avoids modifying system Python):
   ```bash
   cd open_robo_vision
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   If `pip install -r requirements.txt` fails on `tensorflow==2.11.1` (not available on Python 3.12), install the rest then add TensorFlow separately:
   ```bash
   pip install ttkbootstrap "jax>=0.5" ott-jax==0.2 ml-collections six pyyaml flax msgpack rich google protobuf einops wrapt requests matplotlib ftfy regex scikit-learn pandas seaborn opencv-python
   pip install tensorflow
   ```

3. **Run the UI from the macOS Terminal** (not from Cursor’s terminal, so the window appears):
   ```bash
   cd /path/to/open_robo_vision
   source .venv/bin/activate
   export PYTHONPATH="$PWD"
   python ui/main_ui.py
   ```
   Or in one line from the project folder:
   ```bash
   cd open_robo_vision && source .venv/bin/activate && export PYTHONPATH="$PWD" && python ui/main_ui.py
   ```

## Docker

Two Compose setups are provided:

| File | Use case |
|------|----------|
| [`docker-compose.yml`](docker-compose.yml) | **Linux** with NVIDIA GPU: Tkinter on the host display via X11, JAX+TensorFlow with CUDA. |
| [`docker-compose.mac.yml`](docker-compose.mac.yml) | **macOS development** (Docker Desktop): CPU-only JAX, UI via **noVNC** in the browser (GPU is not available in Docker on Mac). |

### Model weights

The UI loads a **local** OWL-ViT checkpoint (see `detection/OWL_VIT_v2/owl_vit/configs/owl_v2_clip_b16.py`). **`OWL_CHECKPOINT_PATH`** must be the path *inside the container* to the inner Flax checkpoint directory.

Compose mounts the repo at **`.:/app`**, so weights that live on the host under `detection/OWL_VIT_v2/owl_vit/weights/...` are visible at **`/app/detection/OWL_VIT_v2/owl_vit/weights/...`**—that is the **default** `OWL_CHECKPOINT_PATH` in both compose files (canonical OWL2 B/16 ensemble checkpoint).

Alternatively, put checkpoints under **`./weights`** on the host, keep the default **`OWL_WEIGHTS_DIR`** mount to `/weights/owl`, and set **`OWL_CHECKPOINT_PATH=/weights/owl`** in `.env`.

```bash
cp .env.example .env   # optional; override OWL_CHECKPOINT_PATH / OWL_WEIGHTS_DIR if needed
```

**Docker builds:** [`.dockerignore`](.dockerignore) excludes `detection/OWL_VIT_v2/owl_vit/weights/` and `./weights/` so checkpoints are not sent in the **build context** (saves disk). The **run-time** bind mount `.:/app` still exposes those folders from your working tree.

If Docker reports **no space left on device**, free space with **Docker Desktop → Settings → Resources** (increase disk image) or run `docker system prune -a` (removes unused images/containers).

### Linux (GPU + host window)

**Requirements:** [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) so `docker compose` can use `gpus: all`.

Allow the container to connect to your X server (run once per login; example for local connections):

```bash
xhost +local:docker
export DISPLAY="${DISPLAY:-:0}"
docker compose up --build
```

The Tkinter window should appear on your current desktop. If nothing shows, confirm `echo $DISPLAY` matches what you pass into Compose (see [`.env.example`](.env.example)).

### macOS (development, browser UI)

Docker Desktop does not expose an NVIDIA GPU to containers, and Apple Silicon has no CUDA—this stack uses the **`cpu`** image target and **noVNC**.

**Recommended:**

```bash
cp .env.example .env   # optional
docker compose -f docker-compose.mac.yml up --build
```

If your Docker Desktop / Buildx ever “builds successfully” but then fails to start with an image error, use the classic builder helper script:

```bash
chmod +x scripts/docker-mac-up.sh
./scripts/docker-mac-up.sh
```

Then open **http://localhost:6080/vnc.html** , click **Connect** (no password). Inference is **CPU-only** and will be slower than a native Linux GPU run.

**Optional (advanced):** You can try pointing `DISPLAY` at **XQuartz** on the host instead of noVNC; this is fragile with Docker Desktop’s VM, so noVNC is the supported path for “`docker compose up` works on Mac.”

### ROS

The default images **do not** install ROS 2. The **Connect to ROS** button expects `ros2` on `PATH` (install ROS on the host or build a separate image that extends this one with `rclpy` / `cv_bridge`).

### Environment variables

| Variable | Meaning |
|----------|---------|
| `OWL_CHECKPOINT_PATH` | Path inside the container to the Flax checkpoint folder (default: in-repo canonical path under `/app/detection/.../weights/...`). |
| `OWL_WEIGHTS_DIR` | Host directory bind-mounted to `/weights/owl` (default `./weights`; optional if you use in-repo weights only). |
| `DISPLAY` | Linux X11 display (e.g. `:0`). |
| `XLA_FLAGS` | Optional JAX/XLA flags. |
