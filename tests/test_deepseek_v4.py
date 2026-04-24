import torch
import pytest

from nanochat.deepseek_v4 import (
    DeepSeekHybridAttention,
    DeepSeekMoE,
    DeepSeekV4NanoChat,
    DeepSeekV4NanoConfig,
    HyperConnectionMixer,
    _rope_dim,
    precompute_yarn_rotary,
)


def tiny_config(**overrides):
    values = dict(
        sequence_len=16,
        vocab_size=257,
        n_layer=2,
        n_head=4,
        n_kv_head=1,
        n_embd=64,
        window_size=8,
        attention_layer_pattern="CSA,HCA",
        csa_compress_ratio=4,
        hca_compress_ratio=8,
        rope_head_dim=8,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=4,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        n_hash_layers=1,
        hc_mult=2,
        hc_sinkhorn_iters=8,
        index_topk=2,
        num_nextn_predict_layers=1,
    )
    values.update(overrides)
    return DeepSeekV4NanoConfig(**values)


def test_deepseek_v4_forward_backward_tiny():
    torch.manual_seed(0)
    cfg = tiny_config()
    model = DeepSeekV4NanoChat(cfg)
    model.init_weights()
    x = torch.randint(0, cfg.vocab_size, (2, cfg.sequence_len))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.sequence_len))

    logits = model(x)
    assert logits.shape == (2, cfg.sequence_len, cfg.vocab_size)

    loss = model(x, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    before = model.transformer.h[-1].moe.gate_bias.clone()
    model.transformer.h[-1].moe.last_expert_counts.copy_(torch.tensor([100.0, 0.0, 0.0, 0.0]))
    model.update_aux_free_balance()
    after = model.transformer.h[-1].moe.gate_bias
    assert not torch.equal(before, after)


def test_gate_bias_moves_after_unbalanced_routing():
    torch.manual_seed(1)
    cfg = tiny_config(
        n_layer=1,
        n_hash_layers=0,
        aux_free_balance_rate=0.25,
        n_shared_experts=0,
    )
    moe = DeepSeekMoE(cfg, layer_idx=0)
    with torch.no_grad():
        moe.gate.weight.zero_()

    x = torch.randn(3, 5, cfg.n_embd)
    input_ids = torch.zeros(3, 5, dtype=torch.long)
    _ = moe(x, input_ids)

    counts = moe.last_expert_counts.clone()
    assert counts.max() > counts.min()

    before = moe.gate_bias.clone()
    moe.update_aux_free_balance()
    after = moe.gate_bias

    assert not torch.allclose(before, after)
    assert torch.allclose(after.sum(), torch.tensor(0.0), atol=1e-6)
    overused = counts > counts.float().mean()
    underused = counts < counts.float().mean()
    assert after[overused].mean() < 0
    assert after[underused].mean() > 0


def test_hyper_connection_sinkhorn_is_approximately_doubly_stochastic():
    torch.manual_seed(2)
    cfg = tiny_config(n_layer=1, hc_mult=4, hc_sinkhorn_iters=8)
    assert cfg.hc_sinkhorn_iters >= 8
    mixer = HyperConnectionMixer(cfg)
    x = torch.randn(2, 7, cfg.hc_mult, cfg.n_embd)

    _, _, residual_mapping = mixer(x)

    rows = residual_mapping.sum(dim=-1)
    cols = residual_mapping.sum(dim=-2)
    ones = torch.ones_like(rows)
    assert torch.allclose(rows, ones, atol=2e-3, rtol=2e-3)
    assert torch.allclose(cols, ones, atol=2e-3, rtol=2e-3)


def test_hash_routing_permutation_has_low_adjacent_vocab_overlap():
    cfg = tiny_config(
        vocab_size=257,
        n_layer=4,
        n_hash_layers=-1,
        n_hash_layers_frac=0.25,
        n_routed_experts=16,
        num_experts_per_tok=4,
    )
    moe = DeepSeekMoE(cfg, layer_idx=0)
    assert moe.hash_routing

    overlaps = []
    for token_id in range(cfg.vocab_size - 1):
        current = set(moe.tid2eid[token_id].tolist())
        adjacent = set(moe.tid2eid[token_id + 1].tolist())
        overlaps.append(len(current & adjacent))

    overlaps = torch.tensor(overlaps, dtype=torch.float32)
    high_overlap_fraction = (overlaps >= cfg.num_experts_per_tok - 1).float().mean()
    assert overlaps.mean() < cfg.num_experts_per_tok / 2
    assert high_overlap_fraction < 0.10


@pytest.mark.parametrize("kind", ["CSA", "HCA", "SWA"])
@pytest.mark.parametrize("seq_len", [1, 3, 8, 9])
def test_hybrid_attention_shapes_for_compressed_and_sliding_window_paths(kind, seq_len):
    torch.manual_seed(3)
    cfg = tiny_config(
        sequence_len=16,
        n_layer=1,
        n_head=2,
        n_kv_head=1,
        n_embd=32,
        attention_layer_pattern=kind,
        csa_compress_ratio=8,
        hca_compress_ratio=8,
        window_size=4,
        rope_head_dim=8,
        q_lora_rank=16,
        o_lora_rank=16,
        o_groups=2,
        index_topk=4,
    )
    attention = DeepSeekHybridAttention(cfg, layer_idx=0)
    bound = 3**0.5 * cfg.n_embd**-0.5
    for parameter in attention.parameters():
        if parameter.ndim > 1:
            torch.nn.init.uniform_(parameter, -bound, bound)
        else:
            torch.nn.init.zeros_(parameter)

    x = torch.randn(2, seq_len, cfg.n_embd)
    cos, sin = precompute_yarn_rotary(cfg.sequence_len, _rope_dim(cfg), cfg)
    y = attention(x, (cos[:, :seq_len], sin[:, :seq_len]))

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    if kind in {"CSA", "HCA"} and seq_len <= attention.compress_ratio:
        assert attention.compressor is not None
        assert attention.compressor(x)[0].size(1) == 1
