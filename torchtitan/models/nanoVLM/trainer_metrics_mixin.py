# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch


def compute_effective_token_ratio(effective_tokens: float, token_capacity: float) -> float:
    token_capacity_value = float(token_capacity)
    if token_capacity_value <= 0:
        # Empty microbatches can appear in edge cases (e.g., uneven last shard); treat as 0% usage.
        return 0.0
    return float(effective_tokens) / token_capacity_value


def extract_images_per_sample(images: Any, fallback_batch_size: int) -> list[int]:
    if isinstance(images, list):
        image_counts: list[int] = []
        for sample_images in images:
            if isinstance(sample_images, list):
                image_counts.append(len(sample_images))
            elif isinstance(sample_images, torch.Tensor):
                if sample_images.ndim == 4:
                    image_counts.append(int(sample_images.shape[0]))
                else:
                    image_counts.append(1)
            else:
                image_counts.append(0)
        return image_counts

    if isinstance(images, torch.Tensor) and images.ndim >= 4:
        return [1 for _ in range(int(images.shape[0]))]

    return [0 for _ in range(max(fallback_batch_size, 0))]


class NanoVLMMetricsMixin:
    def _reset_nanovlm_training_stats_window(self) -> None:
        self._nanovlm_training_stats_window = {
            "tokens_per_second_sum": 0.0,
            "tokens_per_second_count": 0,
            "data_load_time_sum": 0.0,
            "data_load_time_count": 0,
            "data_load_time_max": 0.0,
            "fw_bw_time_sum": 0.0,
            "fw_bw_time_count": 0,
            "fw_bw_time_max": 0.0,
            "post_process_time_sum": 0.0,
            "post_process_time_count": 0,
            "post_process_time_max": 0.0,
            "effective_token_ratio_sum": 0.0,
            "effective_token_ratio_count": 0,
            "images_per_sample_sum": 0.0,
            "images_per_sample_count": 0,
            "images_per_sample_min": float("inf"),
            "images_per_sample_max": 0.0,
        }

    def _dist_reduce_scalar(
        self, value: float, op: torch.distributed.ReduceOp
    ) -> float:
        if not torch.distributed.is_initialized():
            return float(value)
        reduced = torch.tensor(value, dtype=torch.float64, device=self.device)
        torch.distributed.all_reduce(reduced, op=op)
        return float(reduced.item())

    def _update_nanovlm_training_stats_window(
        self,
        *,
        tokens_per_second: float,
        data_load_time: float,
        fw_bw_time: float,
        post_process_time: float,
        effective_token_ratio: float,
        images_per_sample: list[int],
    ) -> None:
        stats = self._nanovlm_training_stats_window
        stats["tokens_per_second_sum"] += float(tokens_per_second)
        stats["tokens_per_second_count"] += 1

        stats["data_load_time_sum"] += float(data_load_time)
        stats["data_load_time_count"] += 1
        stats["data_load_time_max"] = max(
            float(stats["data_load_time_max"]), float(data_load_time)
        )

        stats["fw_bw_time_sum"] += float(fw_bw_time)
        stats["fw_bw_time_count"] += 1
        stats["fw_bw_time_max"] = max(float(stats["fw_bw_time_max"]), float(fw_bw_time))

        stats["post_process_time_sum"] += float(post_process_time)
        stats["post_process_time_count"] += 1
        stats["post_process_time_max"] = max(
            float(stats["post_process_time_max"]), float(post_process_time)
        )

        stats["effective_token_ratio_sum"] += float(effective_token_ratio)
        stats["effective_token_ratio_count"] += 1

        if images_per_sample:
            image_sum = float(sum(images_per_sample))
            image_count = int(len(images_per_sample))
            image_min = float(min(images_per_sample))
            image_max = float(max(images_per_sample))
            stats["images_per_sample_sum"] += image_sum
            stats["images_per_sample_count"] += image_count
            stats["images_per_sample_min"] = min(
                float(stats["images_per_sample_min"]), image_min
            )
            stats["images_per_sample_max"] = max(
                float(stats["images_per_sample_max"]), image_max
            )

    def _build_nanovlm_lr_metrics(self) -> dict[str, float]:
        lr_metrics: dict[str, float] = {}
        name_map = {
            "projector": "lr_mp",
            "vision": "lr_vision_backbone",
            "lm": "lr_language_backbone",
        }

        for optimizer in self.optimizers.optimizers:
            for group_idx, group in enumerate(optimizer.param_groups):
                group_name = group.get("name")
                if group_name is None:
                    group_name = f"group_{group_idx}"
                metric_name = name_map.get(group_name, group_name)
                key = f"training_stats/{metric_name}"
                lr_metrics[key] = float(group["lr"])
        return lr_metrics

    def _build_nanovlm_training_stats_metrics(self, grad_norm: float) -> dict[str, float]:
        stats = self._nanovlm_training_stats_window
        if stats["tokens_per_second_count"] <= 0:
            return {}

        def _global_mean(sum_key: str, count_key: str) -> float:
            global_sum = self._dist_reduce_scalar(
                float(stats[sum_key]),
                torch.distributed.ReduceOp.SUM,
            )
            global_count = self._dist_reduce_scalar(
                float(stats[count_key]),
                torch.distributed.ReduceOp.SUM,
            )
            if global_count <= 0:
                return 0.0
            return global_sum / global_count

        result: dict[str, float] = {
            "training_stats/avg_tokens_per_second": _global_mean(
                "tokens_per_second_sum", "tokens_per_second_count"
            ),
            "training_stats/avg_data_load_time": _global_mean(
                "data_load_time_sum", "data_load_time_count"
            ),
            "training_stats/avg_fw_bw_time": _global_mean(
                "fw_bw_time_sum", "fw_bw_time_count"
            ),
            "training_stats/avg_post_process_time": _global_mean(
                "post_process_time_sum", "post_process_time_count"
            ),
            "training_stats/avg_effective_token_ratio": _global_mean(
                "effective_token_ratio_sum", "effective_token_ratio_count"
            ),
            "training_stats/avg_images_per_sample": _global_mean(
                "images_per_sample_sum", "images_per_sample_count"
            ),
            "training_stats/max_data_load_time": self._dist_reduce_scalar(
                float(stats["data_load_time_max"]),
                torch.distributed.ReduceOp.MAX,
            ),
            "training_stats/max_fw_bw_time": self._dist_reduce_scalar(
                float(stats["fw_bw_time_max"]),
                torch.distributed.ReduceOp.MAX,
            ),
            "training_stats/max_post_process_time": self._dist_reduce_scalar(
                float(stats["post_process_time_max"]),
                torch.distributed.ReduceOp.MAX,
            ),
            "training_stats/max_images_per_sample": self._dist_reduce_scalar(
                float(stats["images_per_sample_max"]),
                torch.distributed.ReduceOp.MAX,
            ),
            "training_stats/grad_norm": float(grad_norm),
        }

        if stats["images_per_sample_count"] > 0:
            result["training_stats/min_images_per_sample"] = self._dist_reduce_scalar(
                float(stats["images_per_sample_min"]),
                torch.distributed.ReduceOp.MIN,
            )

        result.update(self._build_nanovlm_lr_metrics())
        return result

    def _collect_nanovlm_model_extra_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for model_part in self.model_parts:
            part_metrics = getattr(model_part, "_nanovlm_extra_metrics", None)
            if isinstance(part_metrics, dict):
                for key, value in part_metrics.items():
                    metrics[key] = float(value)
            if hasattr(model_part, "_nanovlm_extra_metrics"):
                delattr(model_part, "_nanovlm_extra_metrics")
        return metrics
