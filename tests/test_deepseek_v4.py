import torch

from nanochat.deepseek_v4 import DeepSeekV4NanoChat, DeepSeekV4NanoConfig


def test_deepseek_v4_forward_backward_tiny():
    cfg = DeepSeekV4NanoConfig(
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
        hc_sinkhorn_iters=2,
        index_topk=2,
        num_nextn_predict_layers=1,
    )
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
