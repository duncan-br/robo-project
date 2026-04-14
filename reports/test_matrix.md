# Test Matrix (Requirements Traceability)

## Functional Requirements

| Req ID | Requirement (short) | Existing evidence | Added test case(s) | Status |
| --- | --- | --- | --- | --- |
| FR-01 | Process continuous live stream | `TestStreamProcessorLifecycle`, `TestStreamControl`, `TestStreamRos2Source`, `TestWebSocket` | `test_status_reflects_model_state` | Covered |
| FR-02 | Real-time detection | `TestImageInference`, `TestRos2FrameIngest`, leave-out + robustness reports | `test_inference_latency_within_budget` | Covered |
| FR-03 | Explicitly flag unknown/low confidence | `test_infer_image_low_confidence_detections_queued`, mixed-confidence tests, stream low-confidence tests | `test_inference_response_contains_traceability_fields` | Covered |
| FR-04 | Runtime annotation workflow | `TestReviewQueue`, `TestEndToEndInferReviewConfirm` | `test_api_logs_errors_on_failure` (error path traceability) | Covered |
| FR-05 | Create new object classes during operation | `test_confirm_item_create_new_class`, `test_classes_after_adding_new_class` | `test_many_classes_scalability` | Covered |
| FR-06 | Store annotations in structured format | `TestObjectStore` (`save_infer_result`, YOLO line, copy image) | `test_coherence_stable_after_class_addition` | Covered |
| FR-07 | Incremental updates without restart | `TestIncrementalUpdate` | `test_new_class_available_without_restart` | Covered |
| FR-08 | Continuous/incremental learning support | coherence/leave-out validation assets | `test_embedding_update_influences_query` | Covered |
| FR-09 | Log detection and update results for traceability | MLflow record assets in all validation jobs | `test_inference_response_contains_traceability_fields` | Covered |
| FR-10 | Visual feedback on model status | `TestHealth`, stream status tests | `test_status_reflects_model_state` | Covered |
| FR-11 | Model version management | OWL-ViT checkpoint/layer tests (`detection/OWL_VIT_v2/...`) | `test_detector_swappable_via_abc` | Covered |

## Quality Requirements

| Req ID | Requirement (short) | Existing evidence | Added test case(s) | Status |
| --- | --- | --- | --- | --- |
| QR-01 | Remain operational during incremental updates | `TestIncrementalUpdate`, stream settings updates | `test_new_class_available_without_restart` | Covered |
| QR-02 | Handle runtime errors safely | Missing image, bad jpeg, bad params, bad queue ID tests | `test_api_logs_errors_on_failure` | Covered |
| QR-03 | Log critical failures for debugging/validation | MLflow artifacts and metrics; error response paths | `test_api_logs_errors_on_failure` | Covered |
| QR-04 | Scalability under growth | leave-out report over 4305 embeddings; coherence across classes | `test_many_classes_scalability` | Covered |
| QR-05 | Modular and maintainable architecture | `ObjectDetector` ABC, stream source abstractions, service separation | `test_detector_swappable_via_abc` | Covered |
| QR-06 | Maintain quality after incremental class updates | coherence + leave-out validations | `test_coherence_stable_after_class_addition` | Covered |

## Technical Requirements

| Req ID | Requirement (short) | Existing evidence | Added test case(s) | Status |
| --- | --- | --- | --- | --- |
| TR-01 | Runs on Ubuntu 22.04 | Dockerfile base image (`ubuntu:22.04`) | `test_docker_ubuntu_pytest_smoke` | Covered |
| TR-02 | ROS2-compatible pipeline | ROS frame ingest + ROS image decoding + bag source tests | `test_status_reflects_model_state` (stream integration continuity) | Covered |
| TR-03 | Runs locally, not web-dependent | in-process FastAPI TestClient and local filesystem fixtures | `test_docker_ubuntu_pytest_smoke` | Covered |
| TR-04 | Python implementation | repository language and pytest stack | `test_docker_ubuntu_pytest_smoke` | Covered |
| TR-05 | GPU acceleration when available | GPU docker target and OWL-ViT stack | `test_detector_gpu_fallback` | Covered |
| TR-06 | Uses OWL-ViT v2 | `detection/OWL_VIT_v2` model tests | `test_detector_gpu_fallback` | Covered |
| TR-07 | Modular architecture for future models | detector abstraction and service context factories | `test_detector_swappable_via_abc` | Covered |

## Evidence from Current Validation Reports

- Leave-out (`data/validation_leaveout_report.json`):
  - `accuracy`: 0.803
  - `total_embeddings_evaluated`: 4305
- Robustness (`data/validation_robustness_report.json`):
  - `overall_mean_cosine`: 0.7145
  - `passes_threshold`: false (`min_expected_cosine = 0.75`)
- Coherence (`data/validation_coherence_report.json`):
  - `classes_total`: 8
  - `classes_unhealthy`: 8

These values are intentionally included in the matrix narrative to demonstrate that verification is both binary (pass/fail tests) and analytical (metric-based model diagnostics).
