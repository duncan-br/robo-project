from __future__ import annotations

import uvicorn

from on_device_app.api import create_app
from on_device_app.config import load_app_paths
from on_device_app.services import ServiceContext


def main() -> int:
    ctx = ServiceContext(load_app_paths())
    app = create_app(ctx)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

