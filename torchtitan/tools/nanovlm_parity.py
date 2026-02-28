from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_NANOVLM_STEP_RE = re.compile(
    r"\[TRAIN\]\s+step=(?P<step>\d+).*?step_loss=(?P<loss>[0-9]*\.?[0-9]+)"
)
_NANOVLM_TPS_RE = re.compile(
    r"\[TRAIN\]\s+step=(?P<step>\d+).*?tokens_per_second=(?P<tps>[0-9]*\.?[0-9]+)"
)

_TORCHTITAN_STEP_RE = re.compile(
    r"step:\s*(?P<step>\d+).*?loss:\s*(?P<loss>[0-9]*\.?[0-9]+)"
)
_TORCHTITAN_TPS_RE = re.compile(r"step:\s*(?P<step>\d+).*?tps:\s*(?P<tps>[0-9,]+)")
_TORCHTITAN_OPT_GROUP_RE = re.compile(
    r"Optimizer group '(?P<name>[^']+)':\s+"
    r"(?P<param_count>\d+)\s+params,\s+lr=(?P<lr>[0-9.eE+-]+)"
)

_WANDB_RUN_RE = re.compile(r"wandb\.ai/.*/runs/(?P<run>[a-zA-Z0-9]+)")


@dataclass(frozen=True, slots=True)
class LossSeries:
    values_by_step: dict[int, float]

    @property
    def max_step(self) -> int:
        return max(self.values_by_step, default=0)


@dataclass(frozen=True, slots=True)
class ThroughputSeries:
    values_by_step: dict[int, float]

    def median_excluding_first_step(self) -> float | None:
        values = [
            value
            for step, value in sorted(self.values_by_step.items())
            if step >= 2
        ]
        if not values:
            return None
        return float(median(values))


@dataclass(frozen=True, slots=True)
class ParsedRunLog:
    losses: LossSeries
    throughput: ThroughputSeries
    wandb_run_id: str | None


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _parse_series(
    text: str, *, loss_re: re.Pattern[str], tps_re: re.Pattern[str], tps_group: str
) -> ParsedRunLog:
    loss_by_step: dict[int, float] = {}
    tps_by_step: dict[int, float] = {}

    for match in loss_re.finditer(text):
        step = int(match.group("step"))
        loss = float(match.group("loss"))
        loss_by_step[step] = loss

    for match in tps_re.finditer(text):
        step = int(match.group("step"))
        raw_tps = match.group(tps_group).replace(",", "")
        tps_by_step[step] = float(raw_tps)

    run_match = _WANDB_RUN_RE.search(text)
    run_id = run_match.group("run") if run_match else None

    return ParsedRunLog(
        losses=LossSeries(values_by_step=loss_by_step),
        throughput=ThroughputSeries(values_by_step=tps_by_step),
        wandb_run_id=run_id,
    )


def parse_nanovlm_log(text: str) -> ParsedRunLog:
    return _parse_series(
        text,
        loss_re=_NANOVLM_STEP_RE,
        tps_re=_NANOVLM_TPS_RE,
        tps_group="tps",
    )


def parse_torchtitan_log(text: str) -> ParsedRunLog:
    return _parse_series(
        strip_ansi(text),
        loss_re=_TORCHTITAN_STEP_RE,
        tps_re=_TORCHTITAN_TPS_RE,
        tps_group="tps",
    )


def parse_torchtitan_optimizer_groups(
    text: str,
) -> list[dict[str, str | int | float]]:
    groups: list[dict[str, str | int | float]] = []
    for match in _TORCHTITAN_OPT_GROUP_RE.finditer(strip_ansi(text)):
        groups.append(
            {
                "name": match.group("name"),
                "param_count": int(match.group("param_count")),
                "lr": float(match.group("lr")),
            }
        )
    return groups


def paired_loss_diff(
    baseline: LossSeries, candidate: LossSeries
) -> tuple[list[tuple[int, float, float, float]], dict[str, float | int]]:
    paired: list[tuple[int, float, float, float]] = []
    common_steps = sorted(set(baseline.values_by_step) & set(candidate.values_by_step))
    for step in common_steps:
        baseline_loss = baseline.values_by_step[step]
        candidate_loss = candidate.values_by_step[step]
        abs_diff = abs(candidate_loss - baseline_loss)
        paired.append((step, baseline_loss, candidate_loss, abs_diff))

    if not paired:
        return paired, {
            "steps_compared": 0,
            "mean_abs_diff": 0.0,
            "max_abs_diff": 0.0,
            "step_of_max_abs_diff": 0,
        }

    mean_abs_diff = sum(row[3] for row in paired) / len(paired)
    worst = max(paired, key=lambda row: row[3])
    return paired, {
        "steps_compared": len(paired),
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": worst[3],
        "step_of_max_abs_diff": worst[0],
    }
