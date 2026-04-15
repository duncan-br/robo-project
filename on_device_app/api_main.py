from __future__ import annotations

from pathlib import Path

import uvicorn
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from on_device_app.api import create_app
from on_device_app.config import load_app_paths
from on_device_app.services import ServiceContext


def main() -> int:
    import logging

    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    ctx = ServiceContext(load_app_paths())

    log.info("Pre-loading OWL-ViT model onto %s ...", _device_summary())
    ctx.object_detector()
    log.info("Model ready — starting API server.")

    app = create_app(ctx)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    return 0


def _device_summary() -> str:
    try:
        import jax

        devs = jax.devices()
        if devs and "cuda" in str(devs[0]).lower():
            return f"GPU ({devs[0]})"
        return f"CPU ({devs[0]})"
    except Exception:
        return "unknown device"


if __name__ == "__main__":
    raise SystemExit(main())

