import pytest
import torch

from mini_wam.models.act import (
    ActionPolicy, LatentEncoder, PolicyDecoder, TransformerEncoderBlock,
    VisualEncoder, act_loss, kl_divergence, masked_l1_loss,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_policy_decoder_shape_and_gradients():
    decoder = PolicyDecoder(d_model=32, nhead=4)
    x = torch.randn(2, 11, 32, requires_grad=True)

    output = decoder(x)

    assert output.shape == (2, 16, 32)
    output.square().mean().backward()
    assert x.grad is not None
    assert decoder.action_embeddings.weight.grad is not None
    assert decoder.decoder_query.weight.grad is not None
    assert decoder.decoder_kv.weight.grad is not None


def test_policy_decoder_rejects_incompatible_head_count():
    with pytest.raises(ValueError, match="nhead must be positive"):
        PolicyDecoder(d_model=32, nhead=0)

    with pytest.raises(ValueError, match="must be divisible"):
        PolicyDecoder(d_model=30, nhead=8)


def test_policy_decoder_checks_memory_shape():
    decoder = PolicyDecoder(d_model=32, nhead=4)

    with pytest.raises(ValueError, match=r"\[B, S, D\]"):
        decoder(torch.randn(2, 32))

    with pytest.raises(ValueError, match="feature dimension"):
        decoder(torch.randn(2, 7, 16))


@pytest.mark.parametrize("memory_length", [11, 38])
def test_policy_decoder_matches_reference_layer_and_gradients(memory_length):
    """Check attention axes, residuals and normalization against PyTorch."""
    torch.manual_seed(7)
    decoder = PolicyDecoder(d_model=32, nhead=4, num_action_queries=16)
    reference = torch.nn.TransformerDecoderLayer(
        d_model=32, nhead=4, dim_feedforward=128,
        dropout=0.0, activation="relu", batch_first=True, norm_first=False,
    )
    with torch.no_grad():
        reference.self_attn.in_proj_weight.copy_(decoder.action_self_attn_qkv.weight)
        reference.self_attn.in_proj_bias.copy_(decoder.action_self_attn_qkv.bias)
        reference.self_attn.out_proj.load_state_dict(decoder.action_self_attn_output.state_dict())
        reference.multihead_attn.in_proj_weight.copy_(torch.cat([
            decoder.decoder_query.weight, decoder.decoder_kv.weight,
        ]))
        reference.multihead_attn.in_proj_bias.copy_(torch.cat([
            decoder.decoder_query.bias, decoder.decoder_kv.bias,
        ]))
        reference.multihead_attn.out_proj.load_state_dict(decoder.cross_attn_output.state_dict())
        reference.linear1.load_state_dict(decoder.ffn[0].state_dict())
        reference.linear2.load_state_dict(decoder.ffn[2].state_dict())
        for name in ("norm1", "norm2", "norm3"):
            getattr(reference, name).load_state_dict(getattr(decoder, name).state_dict())

    memory = torch.randn(2, memory_length, 32, requires_grad=True)
    reference_memory = memory.detach().clone().requires_grad_()
    tokens = decoder.action_embeddings.weight.detach().clone().requires_grad_()
    actual = decoder(memory)
    expected = reference(tokens.unsqueeze(0).expand(2, -1, -1), reference_memory)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    # A nonuniform probe avoids the nearly constant squared norm after LayerNorm.
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(memory.grad, reference_memory.grad, atol=2e-6, rtol=1e-4)
    torch.testing.assert_close(decoder.action_embeddings.weight.grad, tokens.grad,
                               atol=2e-6, rtol=1e-4)
    assert memory.grad.abs().sum() > 0
    for parameter in decoder.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_encoder_matches_reference_and_gradients_with_padding():
    torch.manual_seed(12)
    encoder = TransformerEncoderBlock(d_model=32, nhead=4)
    reference = torch.nn.TransformerEncoderLayer(
        32, 4, dim_feedforward=128, dropout=0.0, batch_first=True,
    )
    with torch.no_grad():
        reference.self_attn.in_proj_weight.copy_(encoder.qkv.weight)
        reference.self_attn.in_proj_bias.copy_(encoder.qkv.bias)
        reference.self_attn.out_proj.load_state_dict(encoder.attn_output.state_dict())
        reference.linear1.load_state_dict(encoder.ffn[0].state_dict())
        reference.linear2.load_state_dict(encoder.ffn[2].state_dict())
        reference.norm1.load_state_dict(encoder.norm1.state_dict())
        reference.norm2.load_state_dict(encoder.norm2.state_dict())
    x = torch.randn(2, 7, 32, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    padding = torch.zeros(2, 7, dtype=torch.bool)
    padding[0, -3:] = True
    actual = encoder(x, padding)
    expected = reference(reference_x, src_key_padding_mask=padding)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, atol=2e-6, rtol=1e-4)
    torch.testing.assert_close(encoder.qkv.weight.grad, reference.self_attn.in_proj_weight.grad,
                               atol=3e-6, rtol=1e-4)


def test_visual_encoder_keeps_spatial_tokens():
    encoder = VisualEncoder(d_model=32, pretrained=False)
    assert encoder(torch.randn(2, 3, 96, 96)).shape == (2, 9, 32)
    with pytest.raises(ValueError, match="image must have shape"):
        encoder(torch.randn(2, 2, 3, 96, 96))


def test_posterior_ignores_padding_but_uses_valid_actions_and_time():
    torch.manual_seed(5)
    encoder = LatentEncoder(32, 4, 2, latent_dim=8, chunk_size=4)
    position = torch.randn(2, 2)
    actions = torch.randn(2, 4, 2, requires_grad=True)
    valid = torch.tensor([[True, True, False, False], [True, True, True, False]])
    _, mu, logvar = encoder(position, actions, valid)
    corrupted = actions.detach().clone().masked_fill(~valid[..., None], float("nan"))
    _, other_mu, other_logvar = encoder(position, corrupted, valid)
    torch.testing.assert_close(mu, other_mu, atol=0, rtol=0)
    torch.testing.assert_close(logvar, other_logvar, atol=0, rtol=0)
    swapped = actions.detach().clone()
    swapped[:, :2] = swapped[:, :2].flip(1)
    _, swapped_mu, _ = encoder(position, swapped, valid)
    assert not torch.allclose(mu, swapped_mu)
    (mu.square().sum() + logvar.square().sum()).backward()
    assert actions.grad[valid].abs().sum() > 0
    assert torch.count_nonzero(actions.grad[~valid]) == 0
    assert "time_encoding" in dict(encoder.named_buffers())
    assert "time_encoding" not in dict(encoder.named_parameters())


def test_losses_match_hand_calculation_and_ignore_nonfinite_padding():
    prediction = torch.tensor([[[1., -3.], [99., 99.]]], requires_grad=True)
    target = torch.tensor([[[0., 0.], [float("nan"), float("inf")]]])
    valid = torch.tensor([[True, False]])
    mu = torch.tensor([[1., 2.]], requires_grad=True)
    logvar = torch.zeros_like(mu, requires_grad=True)
    losses = act_loss(prediction, target, valid, mu, logvar, beta=2.)
    assert losses["action_l1"].item() == 2.
    assert losses["kl"].item() == 2.5
    assert losses["loss"].item() == 7.
    losses["loss"].backward()
    assert torch.count_nonzero(prediction.grad[:, 1]) == 0
    torch.testing.assert_close(mu.grad, 2 * mu.detach())
    assert kl_divergence(torch.zeros(2, 3), torch.zeros(2, 3)).item() == 0


def test_mask_and_empty_sample_validation():
    actions = torch.zeros(2, 4, 2)
    with pytest.raises(TypeError, match="bool"):
        masked_l1_loss(actions, actions, torch.ones(2, 4))
    with pytest.raises(ValueError, match="at least one valid"):
        masked_l1_loss(actions, actions, torch.tensor([[True] * 4, [False] * 4]))
    with pytest.raises(ValueError, match="shape"):
        masked_l1_loss(actions, actions, torch.ones(2, 3, dtype=torch.bool))


def small_policy():
    return ActionPolicy(pretrained=False, d_model=32, nhead=4,
                        num_encoder_layers=1, num_latent_layers=1,
                        latent_dim=8, chunk_size=4)


def test_policy_training_backward_and_optimizer_step():
    torch.manual_seed(3)
    policy = small_policy()
    image, position = torch.randn(2, 3, 96, 96), torch.randn(2, 2)
    target = torch.randn(2, 4, 2)
    valid = torch.tensor([[True, True, False, False], [True] * 4])
    with pytest.raises(ValueError, match="training requires"):
        policy(image, position)
    output = policy(image, position, target, valid)
    assert output["actions"].shape == (2, 4, 2)
    losses = act_loss(output["actions"], target, valid,
                      output["mu"], output["logvar"], beta=0.1)
    losses["loss"].backward()
    for name, parameter in policy.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for name in ("visual_encoder", "state_encoder", "latent_encoder",
                 "latent_projection", "policy_encoder", "policy_decoder", "action_head"):
        assert sum(p.grad.abs().sum() for p in getattr(policy, name).parameters()) > 0
    previous = policy.action_head.weight.detach().clone()
    torch.optim.Adam(policy.parameters(), lr=1e-4).step()
    assert not torch.equal(previous, policy.action_head.weight)


def test_inference_zero_latent_repeatability_and_reload(tmp_path):
    policy = small_policy().eval()
    image, position = torch.randn(1, 3, 96, 96), torch.randn(1, 2)
    latents = []
    hook = policy.latent_projection.register_forward_pre_hook(
        lambda module, inputs: latents.append(inputs[0].detach().clone())
    )
    with torch.no_grad():
        rng_before = torch.get_rng_state()
        expected = policy(image, position)
        assert torch.equal(rng_before, torch.get_rng_state())
        torch.testing.assert_close(policy(image, position), expected, atol=0, rtol=0)
    hook.remove()
    assert all(torch.count_nonzero(z) == 0 for z in latents)
    with pytest.raises(ValueError, match="observations only"):
        policy(image, position, torch.randn(1, 4, 2))
    path = tmp_path / "act.pt"
    torch.save(policy.state_dict(), path)
    restored = small_policy().eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(restored(image, position), expected, atol=0, rtol=0)
