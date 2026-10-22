"""Vector-space merge primitives: trimming, elect-sign, DARE and the dispatcher."""

from __future__ import annotations

import pytest
import torch

from zeroai.merge.algorithms import (
    WORK_DTYPE,
    bernoulli_drop,
    combine,
    dare_rescale,
    elect_sign,
    interference_energy,
    linear_combine,
    magnitude_trim,
    ties_combine,
)
from zeroai.schemas import MergeMethod


def vec(*values) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.float32)


# ------------------------------------------------------------------- trimming
def test_work_dtype_is_float32():
    assert WORK_DTYPE is torch.float32


def test_magnitude_trim_with_full_density_is_a_clone():
    tensor = vec(1.0, -5.0, 3.0, -2.0)
    out = magnitude_trim(tensor, 1.0)
    assert torch.equal(out, tensor)
    assert out is not tensor


def test_magnitude_trim_keeps_the_largest_coefficients():
    tensor = vec(1.0, -5.0, 3.0, -2.0)
    out = magnitude_trim(tensor, 0.5)
    assert torch.allclose(out, vec(0.0, -5.0, 3.0, 0.0))


def test_magnitude_trim_never_keeps_zero_coefficients():
    tensor = vec(0.0, 0.0, 4.0, -1.0)
    out = magnitude_trim(tensor, 0.25)
    assert torch.allclose(out, vec(0.0, 0.0, 4.0, 0.0))


def test_magnitude_trim_of_zero_density_returns_zeros():
    out = magnitude_trim(vec(1.0, 2.0, 3.0), 0.0)
    assert torch.allclose(out, torch.zeros(3))


def test_magnitude_trim_of_an_empty_tensor_is_safe():
    tensor = torch.zeros(0)
    assert magnitude_trim(tensor, 0.5).numel() == 0


def test_magnitude_trim_restores_the_requested_norm():
    tensor = torch.randn(64, generator=torch.Generator().manual_seed(3))
    for norm in ("l1", "l2", "linf"):
        trimmed = magnitude_trim(tensor, 0.4, rescale_norm=norm)
        if norm == "l1":
            assert float(trimmed.abs().sum()) == pytest.approx(float(tensor.abs().sum()), rel=1e-4)
        elif norm == "l2":
            assert float(trimmed.norm()) == pytest.approx(float(tensor.norm()), rel=1e-4)
        else:
            assert float(trimmed.abs().max()) == pytest.approx(float(tensor.abs().max()), rel=1e-4)


def test_magnitude_trim_without_rescale_loses_energy():
    tensor = torch.randn(64, generator=torch.Generator().manual_seed(4))
    trimmed = magnitude_trim(tensor, 0.4)
    assert float(trimmed.norm()) < float(tensor.norm())


def test_magnitude_trim_rejects_an_unknown_norm():
    with pytest.raises(NotImplementedError):
        magnitude_trim(vec(1.0, 2.0), 0.5, rescale_norm="l3")


def test_magnitude_trim_preserves_float16_storage_dtype():
    tensor = vec(1.0, -5.0, 3.0, -2.0).to(torch.float16)
    assert magnitude_trim(tensor, 0.5).dtype is torch.float16


# ----------------------------------------------------------------- bernoulli
def test_bernoulli_drop_with_full_density_is_a_clone():
    tensor = vec(1.0, 2.0, 3.0, 4.0)
    assert torch.equal(bernoulli_drop(tensor, 1.0), tensor)


def test_bernoulli_drop_is_deterministic_for_a_seeded_generator():
    tensor = torch.randn(128, generator=torch.Generator().manual_seed(11))
    gen_a = torch.Generator().manual_seed(99)
    gen_b = torch.Generator().manual_seed(99)
    out_a = bernoulli_drop(tensor, 0.5, generator=gen_a)
    out_b = bernoulli_drop(tensor, 0.5, generator=gen_b)
    assert torch.equal(out_a, out_b)


def test_bernoulli_drop_keeps_the_requested_density_of_coefficients():
    tensor = torch.ones(2000)
    gen = torch.Generator().manual_seed(5)
    kept = int((bernoulli_drop(tensor, 0.4, rescale_norm=None, generator=gen) != 0).sum())
    assert 700 <= kept <= 900


def test_bernoulli_drop_rescales_to_the_original_l2_norm_by_default():
    tensor = torch.randn(256, generator=torch.Generator().manual_seed(12))
    dropped = bernoulli_drop(tensor, 0.5)
    assert float(dropped.norm()) == pytest.approx(float(tensor.norm()), rel=1e-3)


def test_bernoulli_drop_upcasts_float16_but_returns_the_original_dtype():
    tensor = torch.randn(64, generator=torch.Generator().manual_seed(6)).to(torch.float16)
    out = bernoulli_drop(tensor, 0.5, generator=torch.Generator().manual_seed(1))
    assert out.dtype is torch.float16


def test_bernoulli_drop_of_an_empty_tensor_is_safe():
    assert bernoulli_drop(torch.zeros(0), 0.5).numel() == 0


# --------------------------------------------------------------- elect sign
def test_elect_sign_takes_the_weighted_majority():
    out = elect_sign([vec(1.0, -1.0, 1.0), vec(-1.0, 1.0, 1.0)], [2.0, 1.0])
    assert torch.allclose(out, vec(1.0, -1.0, 1.0))


def test_elect_sign_resolves_ties_towards_positive():
    out = elect_sign([vec(1.0, -1.0), vec(-1.0, 1.0)], [1.0, 1.0])
    assert torch.allclose(out, vec(1.0, 1.0))


def test_elect_sign_requires_at_least_one_tensor():
    with pytest.raises(ValueError):
        elect_sign([], [])


def test_elect_sign_output_is_binary_and_float32():
    out = elect_sign([torch.randn(32, generator=torch.Generator().manual_seed(2))], [1.0])
    assert out.dtype is WORK_DTYPE
    assert set(out.tolist()) <= {-1.0, 1.0}


# ------------------------------------------------------------- linear combine
def test_linear_combine_is_a_weighted_sum():
    out = linear_combine([vec(1.0, 2.0), vec(3.0, 4.0)], [0.5, 0.25])
    assert torch.allclose(out, vec(1.25, 2.0))


def test_linear_combine_rejects_length_mismatch():
    with pytest.raises(ValueError):
        linear_combine([vec(1.0)], [1.0, 2.0])


def test_linear_combine_rejects_an_empty_list():
    with pytest.raises(ValueError):
        linear_combine([], [])


def test_linear_combine_accumulates_in_float32_even_for_float16_inputs():
    out = linear_combine([vec(1.0, 2.0).to(torch.float16)], [1.0])
    assert out.dtype is WORK_DTYPE


# ------------------------------------------------------------------ ties
def test_ties_combine_discards_sign_outliers():
    # The weighted-majority direction wins; the minority vector keeps only the
    # coefficients that agree with the elected sign.
    out = ties_combine([vec(2.0, -2.0), vec(-1.0, 1.0)], [1.0, 1.0], density=1.0)
    assert torch.allclose(out, vec(2.0, -2.0))


def test_ties_combine_keeps_both_sides_of_an_exact_tie():
    # Zero-sum votes resolve towards +1, so neither contributor is discarded.
    out = ties_combine([vec(1.0, -1.0, 1.0), vec(-1.0, 1.0, 1.0)], [1.0, 1.0], density=1.0)
    assert torch.allclose(out, vec(1.0, 1.0, 2.0))


def test_ties_combine_follows_the_weighted_majority():
    out = ties_combine([vec(1.0, 1.0), vec(-3.0, -3.0)], [1.0, 1.0], density=1.0)
    assert torch.allclose(out, vec(-3.0, -3.0))


def test_ties_combine_trims_before_electing():
    a = vec(0.01, 5.0, -0.02)
    b = vec(4.0, -0.03, 0.02)
    out = ties_combine([a, b], [1.0, 1.0], density=0.5)
    # Only the dominant coefficient of each vector survives trimming, so the
    # tiny opposing coefficients can never flip a consensus sign.
    assert out[0] == pytest.approx(4.0)
    assert out[1] == pytest.approx(5.0)
    assert out[2] == pytest.approx(0.0)


# ------------------------------------------------------------------ dare
def test_dare_rescale_with_no_drop_only_applies_the_scale():
    tensor = vec(1.0, 2.0, 3.0)
    out = dare_rescale(tensor, keep_density=1.0, scale=2.5)
    assert torch.allclose(out, tensor * 2.5)


def test_dare_rescale_restores_the_norm_before_scaling():
    tensor = torch.randn(256, generator=torch.Generator().manual_seed(21))
    out = dare_rescale(tensor, keep_density=0.3, scale=1.7, generator=torch.Generator().manual_seed(8))
    assert float(out.norm()) == pytest.approx(1.7 * float(tensor.norm()), rel=1e-2)


# -------------------------------------------------------------- dispatcher
def test_combine_linear_is_the_weighted_sum():
    out = combine(MergeMethod.LINEAR, [vec(1.0, 2.0), vec(3.0, 4.0)], [1.0, 0.5])
    assert torch.allclose(out, vec(2.5, 4.0))


def test_combine_accepts_a_method_name_string():
    out = combine("linear", [vec(1.0)], [2.0])
    assert torch.allclose(out, vec(2.0))


def test_combine_rejects_an_unknown_method():
    with pytest.raises(ValueError):
        combine("magic", [vec(1.0)], [1.0])


def test_combine_requires_at_least_one_task_vector():
    with pytest.raises(ValueError):
        combine(MergeMethod.LINEAR, [], [])


def test_combine_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        combine(MergeMethod.LINEAR, [vec(1.0)], [1.0, 2.0])


def test_combine_taskslim_trims_without_sign_election():
    a = vec(0.5, 5.0)
    b = vec(4.0, -0.5)
    out = combine(MergeMethod.TASKSLIM, [a, b], [1.0, 1.0], density=0.5)
    assert torch.allclose(out, vec(4.0, 5.0))


def test_combine_ties_elects_signs():
    out = combine(
        MergeMethod.TIES, [vec(1.0, -1.0, 1.0), vec(-1.0, 1.0, 1.0)], [1.0, 1.0], density=1.0
    )
    assert torch.allclose(out, vec(1.0, 1.0, 2.0))


def test_combine_dare_linear_scales_without_sign_election():
    deltas = [vec(1.0, 2.0), vec(3.0, 4.0)]
    out = combine(
        MergeMethod.DARE_LINEAR,
        deltas,
        [1.0, 1.0],
        dare_keep=1.0,
        dare_scale=2.0,
    )
    assert torch.allclose(out, 2.0 * linear_combine(deltas, [1.0, 1.0]))


def test_combine_dare_ties_drops_then_elects_signs():
    deltas = [vec(1.0, -1.0, 1.0), vec(-1.0, 1.0, 1.0)]
    out = combine(
        MergeMethod.DARE_TIES,
        deltas,
        [1.0, 1.0],
        density=1.0,
        dare_keep=1.0,
        dare_scale=1.0,
        dare_norm=None,
        generator=torch.Generator().manual_seed(4),
    )
    # With no dropping and no rescaling this degenerates to plain TIES.
    assert torch.allclose(out, vec(1.0, 1.0, 2.0))


def test_combine_dare_drop_is_applied_to_every_task_vector():
    tensor = torch.ones(400)
    out = combine(
        MergeMethod.DARE_LINEAR,
        [tensor, tensor],
        [1.0, 0.0],
        dare_keep=0.25,
        dare_norm=None,
        generator=torch.Generator().manual_seed(13),
    )
    kept = int((out != 0).sum())
    assert 60 <= kept <= 140


def test_combine_returns_float32_for_low_precision_inputs():
    out = combine(MergeMethod.LINEAR, [vec(1.0).to(torch.float16)], [1.0])
    assert out.dtype is WORK_DTYPE


# ------------------------------------------------------- interference metric
def test_interference_energy_of_aligned_contributions_is_positive():
    v = vec(1.0, 2.0, 3.0)
    cross, own = interference_energy([v, v])
    assert cross == pytest.approx(float((v * v).sum()))
    assert own == pytest.approx(2.0 * float((v * v).sum()))


def test_interference_energy_of_conflicting_contributions_is_negative():
    v = vec(1.0, 2.0, 3.0)
    cross, own = interference_energy([v, -v])
    assert cross < 0.0
    assert own == pytest.approx(2.0 * float((v * v).sum()))


def test_interference_energy_of_orthogonal_contributions_is_zero():
    cross, own = interference_energy([vec(1.0, 0.0), vec(0.0, 1.0)])
    assert cross == pytest.approx(0.0, abs=1e-6)
    assert own == pytest.approx(2.0)


def test_interference_energy_of_an_empty_list():
    assert interference_energy([]) == (0.0, 0.0)


# ------------------------------------------------- seeded DARE on a cuda tensor
HAS_CUDA = torch.cuda.is_available()


@pytest.mark.skipif(not HAS_CUDA, reason="requires a CUDA device")
def test_bernoulli_drop_accepts_a_cpu_generator_on_a_cuda_tensor():
    """torch rejects a cpu generator for cuda tensors - DARE must not break."""
    tensor = torch.ones(4000, device="cuda")
    out = bernoulli_drop(tensor, 0.4, generator=torch.Generator(device="cpu").manual_seed(7))
    assert out.device.type == "cuda"
    assert out.dtype is torch.float32
    kept = int((out != 0).sum())
    assert 1200 <= kept <= 2800


@pytest.mark.skipif(not HAS_CUDA, reason="requires a CUDA device")
def test_dare_drop_pattern_is_the_same_on_cpu_and_cuda():
    values = torch.ones(4000)
    cpu_out = bernoulli_drop(
        values, 0.4, rescale_norm=None, generator=torch.Generator().manual_seed(9)
    )
    cuda_out = bernoulli_drop(
        values.to("cuda"), 0.4, rescale_norm=None, generator=torch.Generator().manual_seed(9)
    )
    assert torch.equal((cpu_out != 0), (cuda_out != 0).cpu())


@pytest.mark.skipif(not HAS_CUDA, reason="requires a CUDA device")
def test_combine_dare_methods_run_on_cuda_with_a_cpu_generator():
    deltas = [torch.ones(2000, device="cuda"), -torch.ones(2000, device="cuda")]
    for method in (MergeMethod.DARE_LINEAR, MergeMethod.DARE_TIES):
        out = combine(
            method,
            deltas,
            [0.6, 0.4],
            density=0.5,
            dare_keep=0.6,
            dare_scale=1.5,
            generator=torch.Generator(device="cpu").manual_seed(3),
        )
        assert out.device.type == "cuda"
        assert torch.isfinite(out).all()