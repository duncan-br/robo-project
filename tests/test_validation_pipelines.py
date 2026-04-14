from __future__ import annotations

import numpy as np

from improved_pipelines.validate_coherence import (
    assess_class_health_df,
    compute_inter_class_separation_df,
    compute_intra_class_similarity_df,
)


class TestValidationPipelineGaps:
    def test_coherence_stable_after_class_addition(self):
        """Existing intra-class similarity should stay unchanged when a new class is added."""
        base = {
            "class_a": [
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.98, 0.02, 0.0], dtype=np.float32),
                np.array([0.99, 0.01, 0.0], dtype=np.float32),
            ],
            "class_b": [
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                np.array([0.01, 0.99, 0.0], dtype=np.float32),
                np.array([0.02, 0.98, 0.0], dtype=np.float32),
            ],
        }
        base_intra = compute_intra_class_similarity_df(base).set_index("class_name")

        expanded = dict(base)
        expanded["class_c"] = [
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            np.array([0.0, 0.05, 0.95], dtype=np.float32),
            np.array([0.03, 0.0, 0.97], dtype=np.float32),
        ]
        expanded_intra = compute_intra_class_similarity_df(expanded).set_index("class_name")
        inter_df = compute_inter_class_separation_df(expanded)
        health_df = assess_class_health_df(
            expanded_intra.reset_index(),
            inter_df,
            min_exemplars=3,
            min_intra_sim=0.55,
            max_inter_sim=0.9,
        )

        for cname in ("class_a", "class_b"):
            diff = abs(
                float(base_intra.loc[cname, "mean_cosine_sim"])
                - float(expanded_intra.loc[cname, "mean_cosine_sim"])
            )
            assert diff <= 1e-6
        assert {"class_a", "class_b", "class_c"} == set(health_df["class_name"].tolist())
