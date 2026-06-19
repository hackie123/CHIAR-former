import math
def compute_model_flops(B=1, T=256, d=1024, n_heads=16, routing_mode="threshold"):
    attn_flops = B*T*(3*d*d + T*d + T*d + d*d)
    dct_flops  = B*T*d*math.log2(d)*2
    baseline   = 28*attn_flops
    chiar_attn = 0.5*26*attn_flops + attn_flops
    chiar_dct  = dct_flops + 0.5*26*dct_flops
    return {
        "baseline_total": baseline,
        "chiar_total": chiar_attn + chiar_dct,
        "attn_reduction_pct": round((1 - chiar_attn/baseline)*100, 1),
        "total_reduction_pct": round((1 - (chiar_attn+chiar_dct)/baseline)*100, 1),
    }
