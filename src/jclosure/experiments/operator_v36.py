"""One-token, architecture-native factor extraction and prospective REC read prediction.

The installed kernels, not the symbolic formulas, remain the reference. Factor
extraction is response-blind; native mixer calls are used only after prediction.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from jclosure.cache_v7 import clone_hybrid_cache


def _qwen_factors(module, h, cache):
    from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm
    b, t, _ = h.shape
    if t != 1:
        raise RuntimeError("V36 supports one-token cached decode only")
    mixed = module.in_proj_qkv(h).transpose(1, 2)
    conv = cache.layers[module.layer_idx].conv_states.detach().clone()
    mixed = module.causal_conv1d_update(mixed, conv, module.conv1d.weight.squeeze(1),
                                        module.conv1d.bias, module.activation).transpose(1, 2)
    query, key, value = torch.split(mixed, [module.key_dim, module.key_dim, module.value_dim], dim=-1)
    query = query.reshape(b, t, -1, module.head_k_dim)
    key = key.reshape(b, t, -1, module.head_k_dim)
    value = value.reshape(b, t, -1, module.head_v_dim)
    beta = module.in_proj_b(h).sigmoid()
    a = module.in_proj_a(h)
    g = -module.A_log.float().exp() * F.softplus(a.float() + module.dt_bias)
    if module.num_v_heads // module.num_k_heads > 1:
        repeats = module.num_v_heads // module.num_k_heads
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)
    q, k, v, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32)
                         for x in (query, key, value, beta, g)]
    q = q[:, :, 0] * (1 / (q.shape[-1] ** .5))
    return {"q": q, "k": k[:, :, 0], "v": v[:, :, 0],
            "beta": beta[:, :, 0], "decay": g[:, :, 0].exp(),
            "postconv": mixed.detach().clone(), "gate": module.in_proj_z(h).reshape(b, t, -1, module.head_v_dim)}


def _qwen_parts(f, state):
    s = state.to(f["v"])
    decayed = s * f["decay"].unsqueeze(-1).unsqueeze(-1)
    memory = (decayed * f["k"].unsqueeze(-1)).sum(dim=-2)
    delta = (f["v"] - memory) * f["beta"].unsqueeze(-1)
    update = f["k"].unsqueeze(-1) * delta.unsqueeze(-2)
    new = decayed + update
    raw = (new * f["q"].unsqueeze(-1)).sum(dim=-2)
    return {"decayed": decayed, "memory": memory, "update": update,
            "new_state": new, "raw": raw.reshape(-1, raw.shape[-1]).to(f["postconv"].dtype)}


def _falcon_factors(module, h, cache):
    from transformers.models.falcon_h1.modeling_falcon_h1 import apply_mask_to_padding_states
    b, t, _ = h.shape
    if t != 1:
        raise RuntimeError("V36 supports one-token cached decode only")
    x = apply_mask_to_padding_states(h, None) * module.ssm_in_multiplier
    projected = module.in_proj(x) * module.mup_vector
    gate, conv_input, dt = projected.split([module.intermediate_size, module.conv_dim, module.num_heads], dim=-1)
    work = clone_hybrid_cache(cache)
    conv_states = work.update_conv_state(conv_input.transpose(1, 2), module.layer_idx)
    conv_states = conv_states.to(device=module.conv1d.weight.device)
    post = torch.sum(conv_states * module.conv1d.weight.squeeze(1), dim=-1)
    if module.use_conv_bias:
        post = post + module.conv1d.bias
    post = module.act(post)
    hidden, B, C = torch.split(post, [module.intermediate_size,
                                     module.n_groups * module.ssm_state_size,
                                     module.n_groups * module.ssm_state_size], dim=-1)
    dt = dt[:, 0, :][:, None, ...]
    dt = dt.transpose(1, 2).expand(b, dt.shape[-1], module.head_dim)
    bias = module.dt_bias[..., None].expand(module.dt_bias.shape[0], module.head_dim)
    dt = F.softplus(dt + bias.to(dt.dtype))
    dt = torch.clamp(dt, module.time_step_limit[0], module.time_step_limit[1])
    A = -torch.exp(module.A_log.float())
    A = A[..., None, None].expand(module.num_heads, module.head_dim, module.ssm_state_size).to(torch.float32)
    dA = torch.exp(dt[..., None] * A).to(device=cache.layers[module.layer_idx].recurrent_states.device)
    B = B.reshape(b, module.n_groups, -1)[..., None, :]
    B = B.expand(b, module.n_groups, module.num_heads // module.n_groups, B.shape[-1]).contiguous().reshape(b, -1, B.shape[-1])
    C = C.reshape(b, module.n_groups, -1)[..., None, :]
    C = C.expand(b, module.n_groups, module.num_heads // module.n_groups, C.shape[-1]).contiguous().reshape(b, -1, C.shape[-1])
    hidden = hidden.reshape(b, -1, module.head_dim)
    dBx = (dt[..., None] * B[..., None, :] * hidden[..., None]).to(dA.device)
    D = module.D[..., None].expand(module.D.shape[0], module.head_dim)
    return {"postconv": post, "gate": gate, "dt": dt, "dA": dA, "B": B, "C": C,
            "hidden": hidden, "dBx": dBx, "D": D}


def _falcon_parts(f, state):
    old = state * f["dA"]
    new = old + f["dBx"]
    c = f["C"]
    b, heads, dim, n = new.shape
    read = torch.bmm(new.to(device=c.device, dtype=c.dtype).reshape(b * heads, dim, n),
                     c.reshape(b * heads, n, 1)).reshape(b, heads, dim)
    skip = f["hidden"] * f["D"]
    raw = (read + skip).to(read.dtype).reshape(b, 1, -1)
    return {"decayed": old, "update": f["dBx"], "new_state": new,
            "state_read": read, "skip": skip, "raw": raw}


def factors(key, module, h, cache):
    return _qwen_factors(module, h, cache) if key == "Q" else _falcon_factors(module, h, cache)


def parts(key, f, state):
    return _qwen_parts(f, state) if key == "Q" else _falcon_parts(f, state)


def predicted_delta(key, f, recipient_state, donor_state):
    """Compute the quantization-aware finite difference without observed output."""
    return parts(key, f, donor_state)["raw"] - parts(key, f, recipient_state)["raw"]


@torch.no_grad()
def native_local(module, h, cache):
    """Run the exact installed mixer with one fixed local input and cloned cache."""
    work = clone_hybrid_cache(cache)
    capture = {}

    def before_norm(_module, args):
        capture["raw"] = args[0].detach().clone()

    def after_norm(_module, _args, out):
        capture["normalized"] = out.detach().clone()

    a = module.norm.register_forward_pre_hook(before_norm)
    b = module.norm.register_forward_hook(after_norm)
    try:
        output = module(hidden_states=h, cache_params=work, attention_mask=None)
    finally:
        a.remove(); b.remove()
    if "raw" not in capture or "normalized" not in capture:
        raise RuntimeError("V36 native local read capture missing")
    return {"raw": capture["raw"], "normalized": capture["normalized"],
            "mixer": output.detach().clone(), "cache": work}
