from scripts.nanovlm_ddp_vs_fsdp_benchmark import (
    RunOutcome,
    _gradient_accumulation_steps,
    _parallelism_flags,
    _pick_family_winner,
    _valid_local_batch_sizes,
    _verdict,
)


def _outcome(
    *,
    family: str,
    ac_mode: str,
    local_batch_size: int,
    elapsed_sec: float,
    peak_mem_mib: int,
    median_tps_excl_step1: float,
    return_code: int = 0,
    timed_out: bool = False,
    oom_detected: bool = False,
    max_step: int = 20,
    steps_target: int = 20,
) -> RunOutcome:
    return RunOutcome(
        config_name="cfg",
        family=family,
        ac_mode=ac_mode,
        local_batch_size=local_batch_size,
        global_batch_size=64,
        gradient_accumulation_steps=64 // (local_batch_size * 2),
        steps_target=steps_target,
        return_code=return_code,
        oom_detected=oom_detected,
        timed_out=timed_out,
        elapsed_sec=elapsed_sec,
        peak_mem_mib=peak_mem_mib,
        peak_mem_total_mib=peak_mem_mib * 2,
        max_step=max_step,
        final_loss=5.0,
        median_tps_excl_step1=median_tps_excl_step1,
        log_path="train.log",
        mem_trace_path="mem.csv",
        wandb_run_url=None,
    )


def test_valid_local_batch_sizes_for_two_gpu_global_64():
    assert _valid_local_batch_sizes(global_batch_size=64, nproc_per_node=2) == [
        32,
        16,
        8,
        4,
        2,
        1,
    ]


def test_gradient_accumulation_steps_matches_effective_batch():
    assert _gradient_accumulation_steps(
        global_batch_size=64,
        local_batch_size=8,
        nproc_per_node=2,
    ) == 4
    assert _gradient_accumulation_steps(
        global_batch_size=64,
        local_batch_size=16,
        nproc_per_node=2,
    ) == 2


def test_parallelism_flags_match_ddp_and_fsdp_families():
    assert (
        _parallelism_flags("ddp", 2)
        == "--parallelism.data_parallel_replicate_degree 2 --parallelism.data_parallel_shard_degree 1"
    )
    assert (
        _parallelism_flags("fsdp", 2)
        == "--parallelism.data_parallel_replicate_degree 1 --parallelism.data_parallel_shard_degree 2"
    )


def test_pick_family_winner_prefers_lower_elapsed_time():
    slower = _outcome(
        family="fsdp",
        ac_mode="none",
        local_batch_size=16,
        elapsed_sec=120.0,
        peak_mem_mib=28000,
        median_tps_excl_step1=22000.0,
    )
    faster = _outcome(
        family="fsdp",
        ac_mode="full",
        local_batch_size=8,
        elapsed_sec=110.0,
        peak_mem_mib=29000,
        median_tps_excl_step1=21000.0,
    )
    assert _pick_family_winner([slower, faster]) == faster


def test_pick_family_winner_skips_failed_candidates():
    failed = _outcome(
        family="ddp",
        ac_mode="none",
        local_batch_size=32,
        elapsed_sec=50.0,
        peak_mem_mib=30000,
        median_tps_excl_step1=10000.0,
        return_code=1,
        oom_detected=True,
        max_step=3,
    )
    valid = _outcome(
        family="ddp",
        ac_mode="full",
        local_batch_size=8,
        elapsed_sec=90.0,
        peak_mem_mib=26000,
        median_tps_excl_step1=15000.0,
    )
    assert _pick_family_winner([failed, valid]) == valid


def test_verdict_prefers_fsdp_when_it_wins_speed_and_memory():
    ddp = _outcome(
        family="ddp",
        ac_mode="none",
        local_batch_size=8,
        elapsed_sec=400.0,
        peak_mem_mib=30000,
        median_tps_excl_step1=15000.0,
        steps_target=100,
        max_step=100,
    )
    fsdp = _outcome(
        family="fsdp",
        ac_mode="full",
        local_batch_size=16,
        elapsed_sec=350.0,
        peak_mem_mib=25000,
        median_tps_excl_step1=16000.0,
        steps_target=100,
        max_step=100,
    )
    assert _verdict(ddp=ddp, fsdp=fsdp) == "fsdp_better"


def test_verdict_returns_tradeoff_when_speed_and_memory_split():
    ddp = _outcome(
        family="ddp",
        ac_mode="none",
        local_batch_size=8,
        elapsed_sec=320.0,
        peak_mem_mib=31000,
        median_tps_excl_step1=15000.0,
        steps_target=100,
        max_step=100,
    )
    fsdp = _outcome(
        family="fsdp",
        ac_mode="full",
        local_batch_size=16,
        elapsed_sec=340.0,
        peak_mem_mib=25000,
        median_tps_excl_step1=14000.0,
        steps_target=100,
        max_step=100,
    )
    assert _verdict(ddp=ddp, fsdp=fsdp) == "tradeoff"
