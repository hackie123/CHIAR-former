# utils/flop_counter.py
import math

def compute_model_flops(B=1, T=256, d=256, n_heads=4, R=64, routing_mode="threshold"):
    attn_flops = B*T*(3*d*d + T*d + T*d + d*d)
    dct_flops  = B*T*d*math.log2(d)*2
    baseline   = 4*attn_flops
    chiar_attn = 0.5*attn_flops + 0.5*attn_flops + attn_flops
    chiar_dct  = dct_flops + 0.5*dct_flops + 0.5*dct_flops
    return {
        "baseline_attn_flops": baseline,
        "chiar_attn_flops":    chiar_attn,
        "chiar_total_flops":   chiar_attn + chiar_dct,
        "baseline_total_flops":baseline,
        "attn_reduction_pct":  round((1-chiar_attn/baseline)*100, 1),
    }
