# Tests for GaussianBoltzmannMoE anti-collapse features:
#   - mixing_entropy_gamma (π_e entropy bonus)
#   - centroid_repulsion_lambda (pairwise centroid repulsion)
#   - bias_based_balancing (DeepSeek-V3 aux-loss-free balancing)

import sys
from unittest.mock import MagicMock

# Stub model_conversion submodules that import optional transformers classes
for _name in ("granitemoehybrid", "granitemoeshared"):
    sys.modules.setdefault(
        f"lm_engine.hf_models.model_conversion.{_name}",
        MagicMock(),
    )

import torch
import torch.nn.functional as F
from torch.testing import assert_close

from lm_engine.hf_models.modeling_utils.mlp_blocks.moe_energy import GaussianBoltzmannMoE


def _make_moe(**overrides) -> GaussianBoltzmannMoE:
    """Create a small GaussianBoltzmannMoE for testing."""
    defaults = dict(
        hidden_size=64,
        intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        normalized_topk=True,
        activation_function="gelu_pytorch_tanh",
        add_bias=False,
        dropout=0.0,
        init_method="normal",
        initializer_range=0.02,
        m_width=1.0,
        num_layers=1,
        use_padding_free_transformer=False,
        use_det_normalization=True,
        use_mixing_coefficients=True,
        diversity_lambda=0.0,
        entropy_bonus_gamma=0.0,
        mixing_entropy_gamma=0.0,
        centroid_repulsion_lambda=0.0,
        bias_based_balancing=False,
        bias_update_alpha=0.001,
    )
    defaults.update(overrides)
    return GaussianBoltzmannMoE(**defaults)


class TestMixingEntropyLoss:
    """Tests for the π_e entropy bonus."""

    def test_uniform_mixing_has_max_entropy(self):
        """When mixing_logits are uniform, entropy is maximal → loss is minimal."""
        moe = _make_moe(mixing_entropy_gamma=1.0)
        moe.mixing_logits.data.fill_(0.0)  # uniform π_e
        loss_uniform = moe._compute_mixing_entropy_loss()

        # Skew one expert heavily
        moe.mixing_logits.data = torch.tensor([10.0, -10.0, -10.0, -10.0])
        loss_skewed = moe._compute_mixing_entropy_loss()

        # Uniform should have lower (more negative) loss = higher entropy
        assert loss_uniform < loss_skewed, (
            f"Uniform loss {loss_uniform:.4f} should be < skewed loss {loss_skewed:.4f}"
        )

    def test_mixing_entropy_contributes_to_aux_loss(self):
        """With mixing_entropy_gamma > 0, aux loss should include π_e entropy."""
        moe = _make_moe(mixing_entropy_gamma=0.0).cuda().train()
        x = torch.randn(4, 8, 64, device="cuda")
        _ = moe(x)

        moe2 = _make_moe(mixing_entropy_gamma=1.0).cuda().train()
        moe2.load_state_dict(moe.state_dict(), strict=False)
        _ = moe2(x)
        # Just verify it runs without error — the loss flows through add_aux_loss


class TestCentroidRepulsion:
    """Tests for centroid repulsion loss."""

    def test_collapsed_centroids_high_loss(self):
        """When all centroids are at the same point, repulsion loss should be 0 (maximal)."""
        moe = _make_moe(centroid_repulsion_lambda=1.0)
        # All centroids identical
        moe.centroids.data = torch.zeros(4, 64)
        loss_collapsed = moe._compute_centroid_repulsion_loss()

        # Spread centroids apart
        moe.centroids.data = torch.eye(4, 64) * 10.0
        loss_spread = moe._compute_centroid_repulsion_loss()

        # Collapsed → sq_dist = 0 → loss = 0; spread → sq_dist >> 0 → loss << 0
        assert loss_spread < loss_collapsed, (
            f"Spread loss {loss_spread:.4f} should be < collapsed loss {loss_collapsed:.4f}"
        )

    def test_repulsion_gradient_pushes_centroids_apart(self):
        """Gradient of repulsion loss should push centroids away from each other."""
        moe = _make_moe(centroid_repulsion_lambda=1.0)
        # Two centroids close together
        moe.centroids.data = torch.zeros(4, 64)
        moe.centroids.data[0, 0] = 0.1
        moe.centroids.data[1, 0] = -0.1

        moe.centroids.requires_grad_(True)
        loss = moe._compute_centroid_repulsion_loss()
        loss.backward()

        # Centroid 0 should be pushed in +x direction (away from 1)
        # Centroid 1 should be pushed in -x direction (away from 0)
        assert moe.centroids.grad[0, 0] < 0, "Gradient should push centroid 0 in +x"
        assert moe.centroids.grad[1, 0] > 0, "Gradient should push centroid 1 in -x"


class TestBiasBasedBalancing:
    """Tests for DeepSeek-V3 style bias-based balancing."""

    def test_expert_bias_buffer_exists(self):
        """When enabled, expert_bias should be a registered buffer."""
        moe = _make_moe(bias_based_balancing=True)
        assert hasattr(moe, "expert_bias"), "expert_bias buffer should exist"
        assert moe.expert_bias.shape == (4,)
        assert (moe.expert_bias == 0).all(), "expert_bias should be initialized to zero"

    def test_no_bias_when_disabled(self):
        """When disabled, expert_bias should not exist."""
        moe = _make_moe(bias_based_balancing=False)
        assert not hasattr(moe, "expert_bias"), "expert_bias should not exist when disabled"

    def test_bias_added_to_routing_logits(self):
        """expert_bias should shift routing logits."""
        moe = _make_moe(bias_based_balancing=True)
        h = torch.randn(8, 64)
        energies, _ = moe._compute_energies_and_projections(h)

        logits_before = moe._compute_routing_logits(energies).clone()

        # Manually set a non-zero bias
        moe.expert_bias.fill_(0.0)
        moe.expert_bias[0] = 5.0

        logits_after = moe._compute_routing_logits(energies)

        # Expert 0 logits should increase by 5.0
        diff = logits_after - logits_before
        assert_close(diff[:, 0], torch.full((8,), 5.0), atol=1e-5, rtol=0)
        assert_close(diff[:, 1:], torch.zeros(8, 3), atol=1e-5, rtol=0)

    def test_bias_updates_toward_balance(self):
        """After forward with imbalanced routing, bias should shift to rebalance."""
        moe = _make_moe(bias_based_balancing=True, bias_update_alpha=0.1).cuda().train()

        # Force expert 0 to dominate by setting its centroid to data mean
        moe.centroids.data.zero_()
        moe.centroids.data[0] = torch.zeros(64)  # at origin
        moe.centroids.data[1:] = torch.randn(3, 64) * 100  # far away

        x = torch.randn(4, 16, 64, device="cuda") * 0.1  # data near origin
        _ = moe(x)

        # Expert 0 was overloaded → its bias should have decreased the most
        assert moe.expert_bias[0] == moe.expert_bias.min(), (
            f"Most overloaded expert should have lowest bias, got {moe.expert_bias}"
        )
        # Bias should not all be zero (some update happened)
        assert not (moe.expert_bias == 0).all(), (
            "Bias should have been updated after forward pass"
        )

    def test_bias_is_no_grad(self):
        """expert_bias should not accumulate gradients."""
        moe = _make_moe(bias_based_balancing=True).cuda().train()
        x = torch.randn(4, 8, 64, device="cuda")
        out = moe(x)
        out.sum().backward()
        assert moe.expert_bias.grad is None, "expert_bias should have no gradient"

    def test_bias_saved_in_state_dict(self):
        """expert_bias should be saved/loaded with checkpoints."""
        moe = _make_moe(bias_based_balancing=True)
        moe.expert_bias.fill_(42.0)
        state = moe.state_dict()
        assert "expert_bias" in state

        moe2 = _make_moe(bias_based_balancing=True)
        moe2.load_state_dict(state)
        assert_close(moe2.expert_bias, torch.full((4,), 42.0))


class TestForwardSmoke:
    """Smoke test: all features together, forward + backward works."""

    def test_all_features_enabled(self):
        moe = _make_moe(
            mixing_entropy_gamma=0.1,
            centroid_repulsion_lambda=0.01,
            bias_based_balancing=True,
            bias_update_alpha=0.01,
            diversity_lambda=0.01,
            entropy_bonus_gamma=0.1,
        ).cuda().train()

        x = torch.randn(2, 8, 64, device="cuda", requires_grad=True)
        out = moe(x)
        assert out.shape == x.shape
        out.sum().backward()
        assert x.grad is not None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
