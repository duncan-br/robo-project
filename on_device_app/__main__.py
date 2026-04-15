from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from on_device_app.api_client import ApiClient
from on_device_app.qt import launch_qt


def main() -> int:
    parser = argparse.ArgumentParser(description="Open Robo Vision on-device runtime")
    _ = parser.parse_args()
    api_url = os.environ.get("ON_DEVICE_API_URL", "http://127.0.0.1:8000")
    return launch_qt(ApiClient(base_url=api_url))


if __name__ == "__main__":
    raise SystemExit(main())

