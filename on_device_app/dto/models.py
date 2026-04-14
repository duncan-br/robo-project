from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class InferenceSettings(BaseModel):
    conf_thresh: float = 0.2
    high_conf_min: float = 0.35
    merging_mode: str = "average"
    avg_count: int = 8
    roi_x: float = Field(default=0.0, ge=0.0, le=1.0)
    roi_y: float = Field(default=0.0, ge=0.0, le=1.0)
    roi_w: float = Field(default=1.0, ge=0.0, le=1.0)
    roi_h: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_roi_bounds(self) -> "InferenceSettings":
        if self.roi_x + self.roi_w > 1.0:
            raise ValueError("roi_x + roi_w must be <= 1.0")
        if self.roi_y + self.roi_h > 1.0:
            raise ValueError("roi_y + roi_h must be <= 1.0")
        return self


class DetectionDto(BaseModel):
    class_id: int
    class_name: str
    cx: float
    cy: float
    w: float
    h: float
    score: float
    confidence_level: str


class ReviewItemDto(BaseModel):
    queue_id: str
    image_path: str
    cx: float
    cy: float
    w: float
    h: float
    score: float
    class_id_suggested: int
    class_name_suggested: str


class ConfirmReviewBody(BaseModel):
    class_name: str = Field(min_length=1)
    create_if_missing: bool = False


class InferenceResultDto(BaseModel):
    image_path: str
    pred_count: int
    n_high_saved: int
    n_low_queued: int
    status: str = "ok"


class Ros2FrameIngestResultDto(InferenceResultDto):
    dedup_skipped: int = 0
    detections: list[DetectionDto] = Field(default_factory=list)
    low_confidence_items: list[ReviewItemDto] = Field(default_factory=list)


class QueueListResponse(BaseModel):
    items: list[ReviewItemDto]
    total: int


class StreamFrameMessage(BaseModel):
    frame_jpeg_b64: str
    frame_index: int
    detections: list[DetectionDto]
    low_confidence_items: list[ReviewItemDto]
    stream_fps: float


class StreamStatusDto(BaseModel):
    active: bool
    source_name: str
    frames_processed: int
    current_fps: float


class ImageInferenceRequest(BaseModel):
    image_path: str
    settings: InferenceSettings

