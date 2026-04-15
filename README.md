# OpenRoboVision

Open-source framework for on-device object detection with human-in-the-loop learning. Uses OWL-ViT v2 for open-world detection, Chroma for embedding storage, and a Qt/FastAPI application for runtime inference and annotation review.

## How to Run

All instructions target **Ubuntu 22.04** with CUDA 12.x and Python 3.10.

### Setup

```bash
conda create -n orv_env python=3.10
conda activate orv_env
cd open_robo_vision
pip install -r requirements.txt
```

Copy and configure environment variables:

```bash
cp .env.example .env
```

Set `OWL_CHECKPOINT_PATH` to point to the OWL-ViT v2 Flax checkpoint on disk.

### Object Store

The pipeline expects labeled data in `data/object_store/`:

```
data/object_store/
  classes.txt      # one class name per line (line index = YOLO class_id)
  images/          # image files (e.g. img001.jpg)
  labels/          # YOLO labels (e.g. img001.txt: class_id cx cy w h)
```

Import an existing YOLO dataset:

```bash
python -m improved_pipelines.preload_embeddings \
  --import-yolo /path/to/images /path/to/labels /path/to/classes.txt \
  --object-store data/object_store
```

### Preload Embeddings

Runs OWL-ViT on labeled images and writes vectors into Chroma:

```bash
python -m improved_pipelines.preload_embeddings \
  --object-store data/object_store \
  --chroma-path data/chroma_db
```

Add `--reset-collection` to wipe Chroma first, `--limit N` for a smoke test.

### Application

The on-device application consists of two parts: the **API backend** and the **Qt UI**.

Start the API:

```bash
python -m on_device_app.api_main
```

This runs a FastAPI server on `http://127.0.0.1:8000` (OpenAPI docs at `/docs`). It handles inference, review queue management, and stream processing.

Start the Qt UI (connects to the API):

```bash
python -m on_device_app
```

Override the API URL with `ON_DEVICE_API_URL` if the backend runs elsewhere.

### Docker (GPU)

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
xhost +local:docker
export DISPLAY="${DISPLAY:-:0}"
docker compose up --build
```

### Dagster

Dagster orchestrates the preload and validation pipelines. Start the UI:

```bash
PYTHONPATH=. dagster dev -m dagster_defs.definitions
```

Validate definitions:

```bash
PYTHONPATH=. dagster definitions validate -m dagster_defs.definitions
```

Materialize assets from the CLI:

```bash
PYTHONPATH=. dagster asset materialize -m dagster_defs.definitions --select labeled_image_pairs+
```

Available jobs: `gold_validation_job`, `seed_chroma_job`, `leaveout_validation_job`, `coherence_validation_job`, `robustness_validation_job`.

### MLflow

Validation pipelines log metrics to MLflow (SQLite at `data/mlflow.db` by default).

Start the UI:

```bash
mlflow ui --backend-store-uri sqlite:///data/mlflow.db --allowed-hosts '*'
```

Then open `http://127.0.0.1:5000`. Navigate to **Experiments > Default** to see validation runs. Override the backend with `MLFLOW_TRACKING_URI`. Set it to an empty string to disable MLflow logging.

### Tests

```bash
pytest
```

Configuration is in `pytest.ini`. Coverage reports are written to `reports/`. Minimum coverage threshold is 80%.

### ROS2 Stream Workflow

The system can subscribe to a live ROS2 image topic for real-time inference.

1. Ensure ROS2 Humble is installed and sourced (`rclpy`, `cv_bridge`, `sensor_msgs` must be importable).
2. Start the API backend: `python -m on_device_app.api_main`
3. Start the Qt UI: `python -m on_device_app`
4. In the UI, switch to **ROS2 Live Stream** mode under the Live Inference tab.
5. Enter the topic name (default: `/camera/image_raw`) and click **Start stream**.

The UI subscribes via `on_device_app/ros2/stream_handler.py`, captures frames, and sends them to the API for inference. High-confidence detections are saved automatically; low-confidence ones are queued for manual review in the sidebar.

For **file-based** stream replay (rosbag2 `.db3` or video files), use the **Upload File** mode instead. Upload the file through the UI, then start the stream. The backend reads frames and streams results over a WebSocket.

## Technical Focus Points

### Detector Abstraction (ABC Pattern)

The runtime detector is built around a strict ABC (`ObjectDetector` in `on_device_app/services/detector.py`). The abstract base class enforces `detect()` and `class_names()` on every implementation. The production class `OwlVitV2Detector` wires together the embedding store, the raw OWL-ViT model, and the object store behind factory callbacks so each dependency can be swapped independently. In tests the ABC is satisfied by `FakeDetector`, which avoids loading any model weights at all. This same abstraction is what allows the `StreamProcessor` to accept any detector at construction time, making the entire inference path pluggable without touching the stream or API layers.

### Layered API Architecture

The on-device backend follows a three-layer structure: **endpoints** → **services** → **domain stores**.

- **Endpoints** (`on_device_app/api/app.py`): A single `create_app(ctx)` factory receives a `ServiceContext` and returns a fully configured FastAPI instance. Each route delegates to a service and never contains business logic itself.
- **Services** (`InferenceService`, `ReviewService`, `StreamProcessor`): These encapsulate all decision-making -- confidence routing, dedup, stream lifecycle. They depend only on the `ServiceContext` and its lazy-loaded factories, so they remain testable in isolation.
- **Domain stores** (`ObjectStore`, `ReviewQueue`, `ChromaEmbeddingStore`): Thin wrappers around the filesystem (YOLO label format) and Chroma vector DB. They own persistence and expose simple CRUD operations.

The `ServiceContext` dataclass (`on_device_app/services/context.py`) ties the layers together via thread-safe lazy initialization with `Lock` guards. Each factory (e.g. `object_store()`, `object_detector()`, `stream_processor()`) constructs its resource once and caches it, so hot-path requests never pay initialization cost twice.

### Pydantic DTOs with Cross-Field Validation

All request/response shapes live in `on_device_app/dto/models.py` as Pydantic `BaseModel` classes. `InferenceSettings` uses a `model_validator(mode="after")` to enforce that `roi_x + roi_w <= 1.0` (and the same for Y), rejecting out-of-bounds ROI at the schema level instead of scattering guard clauses through the API. `Ros2FrameIngestResultDto` extends `InferenceResultDto` with dedup and detection fields, keeping the inheritance chain shallow.

### TTL Quantized Box Dedup

`TtlQuantizedBoxDeduper` (`on_device_app/services/dedup.py`) prevents the same object from being re-processed every frame during a live ROS2 stream. It quantizes normalized YOLO box coordinates (cx, cy, w, h) into integer grid cells and keys them with the class ID. A `time.monotonic()` TTL window decides whether the box was already seen. The quantization is coarse enough that the same physical object maps to the same key across minor jitter, but two distinct objects in the same frame produce different keys. A background prune keeps memory bounded to `max_keys`.

### Incremental Chroma Preload with Fingerprinting

`preload_embeddings.py` computes a SHA-256 fingerprint over each image + label file pair. On a second run it queries Chroma's stored fingerprints, diffs against the current manifest, and only re-runs the OWL-ViT detector on new or changed pairs. Unchanged images are skipped entirely, including the model load (the detector is lazily instantiated only when the worklist is non-empty). This makes re-runs near-instant when nothing changed while still catching label edits or new images.

### Validation Pipeline Suite

Four offline validation pipelines measure embedding and model quality from different angles:

- **Gold validation** (`validate_gold.py`): Greedy IoU-matched precision / recall / F1 against ground-truth labels on a random sample of the object store. Uses score-sorted matching so high-confidence predictions are matched first.
- **Leave-out cross-validation** (`validate_leaveout.py`): K-fold split by image ID so all embeddings from one image land in the same fold. A cosine KNN classifier predicts class labels on held-out embeddings without any model forward pass -- pure vector similarity.
- **Coherence analysis** (`validate_coherence.py`): Computes intra-class cosine similarity (compactness) and inter-class centroid cosine similarity (separation), then flags classes that are sparse, low-similarity, or too close to another centroid.
- **Robustness validation** (`validate_robustness.py`): Applies six augmentations (brightness, contrast, Gaussian noise/blur) to sampled images, extracts embeddings from the augmented versions, and compares them against the originals via cosine similarity. Measures how stable the embedding space is under realistic perturbations.

Each pipeline produces a JSON report and optionally logs metrics + artifacts to MLflow.

### Dagster Orchestration

All offline pipelines are wrapped as Dagster assets with Parquet I/O managers, so intermediate DataFrames are materialized to disk and can be inspected or re-used without re-running upstream stages. Resource injection (`ChromaStoreResource`, `OwlDetectorResource`, `MlflowResource`) keeps asset code free of hardcoded paths. A daily schedule and a one-shot sensor (`build_seed_once_sensor`) handle automated execution.

### Stream Processing with Multi-Source Support

`StreamProcessor` (`on_device_app/services/stream_service.py`) runs inference on a background thread and publishes `StreamFrameMessage` DTOs over a WebSocket. It accepts any `StreamSource` (another ABC): `VideoFileSource` wraps OpenCV, `BagFileSource` reads ROS2 bag files (with fallback to `rosbag2_py` or `sqlite3`), and `Ros2TopicSource` subscribes to a live image topic. Inference runs every N-th frame; intermediate frames re-use cached detections to keep the UI responsive without saturating the model.

## Project Structure

```
open_robo_vision/
├── on_device_app/          # On-device runtime (API + Qt UI)
│   ├── api/                # FastAPI application (endpoints for inference, review, streams)
│   ├── qt/                 # Qt widgets (live inference view, review dialog, settings)
│   ├── ros2/               # ROS2 subscriber for live image streams
│   ├── services/           # Business logic (inference, dedup, stream processing, detection)
│   ├── dto/                # Pydantic data transfer objects
│   ├── workers/            # Background inference subprocess
│   ├── api_client.py       # HTTP client for the API
│   ├── api_main.py         # API server entrypoint
│   └── __main__.py         # Qt UI entrypoint
├── improved_pipelines/     # Offline pipelines (preload, inference, validation)
├── dagster_defs/           # Dagster assets, jobs, schedules, and resources
├── detection/              # OWL-ViT v2 model code (third-party, not modified)
├── data/                   # Runtime data (object store, Chroma DB, review queue, reports)
├── tests/                  # Pytest test suite
├── docker-compose.yml      # Linux GPU compose
└── docker-compose.mac.yml  # macOS CPU + noVNC compose
```

### on_device_app

This is the main runtime application, split into an API backend and a Qt desktop UI.

**API** (`on_device_app/api/`): FastAPI server with endpoints for health checks, object class listing, review queue CRUD, image inference, ROS2 frame ingestion, and stream upload/start/stop/status/WebSocket. The API manages all data flow — inference results are routed to the object store (high confidence) or review queue (low confidence).

**Qt UI** (`on_device_app/qt/`): PySide6 desktop interface with tabs for live inference (upload or ROS2 mode), detection visualization with bounding boxes, ROI configuration, and an inline review dialog for low-confidence items.

**Services** (`on_device_app/services/`): Core logic layer. `InferenceService` runs detection through the OWL-ViT model, routes results by confidence. `StreamProcessor` handles continuous frame processing from video/bag/ROS2 sources. `TtlQuantizedBoxDeduper` suppresses duplicate detections across adjacent frames.

**ROS2** (`on_device_app/ros2/`): `RosImageStreamHandler` is a QThread that subscribes to a ROS2 image topic and emits frames to the Qt UI for display and inference.

### improved_pipelines

Offline batch pipelines for embedding management and model validation:

- `preload_embeddings` — seeds Chroma from labeled object store data
- `validate_gold` — precision/recall/F1 against ground truth
- `validate_leaveout` — K-fold cosine KNN cross-validation on embeddings
- `validate_coherence` — intra-class compactness and inter-class separation
- `validate_robustness` — embedding drift under augmentations

### dagster_defs

Dagster code location that wraps `improved_pipelines` into materializable assets with Parquet I/O, MLflow logging, and scheduling. Provides a web UI for pipeline execution and monitoring.
