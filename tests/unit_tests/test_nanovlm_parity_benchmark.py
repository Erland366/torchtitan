from scripts.nanovlm_parity_benchmark import _baseline_can_ignore_return_code


def test_baseline_teardown_crash_is_ignored_after_full_steps():
    assert _baseline_can_ignore_return_code(
        return_code=134,
        parsed_max_step=100,
        steps_target=100,
        log_text=(
            "Fatal Python error: PyGILState_Release\n"
            "Python runtime state: finalizing\n"
        ),
    )


def test_baseline_terminate_without_active_exception_is_ignored():
    assert _baseline_can_ignore_return_code(
        return_code=134,
        parsed_max_step=100,
        steps_target=100,
        log_text="terminate called without an active exception",
    )


def test_nonzero_before_target_steps_is_not_ignored():
    assert not _baseline_can_ignore_return_code(
        return_code=134,
        parsed_max_step=60,
        steps_target=100,
        log_text="Fatal Python error: PyGILState_Release\n",
    )


def test_nonzero_without_known_signature_is_not_ignored():
    assert not _baseline_can_ignore_return_code(
        return_code=1,
        parsed_max_step=100,
        steps_target=100,
        log_text="RuntimeError: out of memory",
    )
