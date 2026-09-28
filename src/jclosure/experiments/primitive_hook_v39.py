"""Falcon one-layer native-factor replacement; all descendants recompute naturally."""
from __future__ import annotations

from contextlib import AbstractContextManager

import torch
import torch.nn.functional as F

from jclosure.experiments.transaction_v28 import thash


FACTOR_KEYS = ("MIXER_INPUT", "X", "B", "C", "DT", "S", "DBX", "OUTPUT_GATE")


def _falcon_step(mixer, input_states, cache_params, attention_mask, replacements, record):
    from transformers.models.falcon_h1.modeling_falcon_h1 import apply_mask_to_padding_states

    def take(name, natural):
        record[name] = natural.detach().clone()
        if name not in replacements:
            return natural
        requested = replacements[name]
        if (requested.shape != natural.shape or requested.dtype != natural.dtype or
                requested.device != natural.device):
            raise RuntimeError(f"V39 primitive factor shape/dtype/device mismatch: {name}")
        realized = requested.detach().clone()
        if not torch.equal(realized, requested):
            raise RuntimeError(f"V39 primitive writeback mismatch: {name}")
        record[name + "_requested_hash"] = thash(requested)
        record[name + "_realized_hash"] = thash(realized)
        record[name + "_native_hash"] = thash(natural)
        record[name + "_exact_writeback"] = True
        return realized

    batch, seq_len, _ = input_states.shape
    if seq_len != 1 or cache_params is None or not cache_params.has_previous_state(mixer.layer_idx):
        raise RuntimeError("V39 primitive hook requires cached one-token decode")
    dtype = input_states.dtype
    input_states = take("MIXER_INPUT", input_states)
    input_states = apply_mask_to_padding_states(input_states, attention_mask)
    input_states = input_states * mixer.ssm_in_multiplier
    projected = mixer.in_proj(input_states) * mixer.mup_vector
    gate, conv_input, dt = projected.split(
        [mixer.intermediate_size, mixer.conv_dim, mixer.num_heads], dim=-1)
    conv_input = conv_input.transpose(1, 2)
    conv_states = cache_params.update_conv_state(conv_input, mixer.layer_idx)
    conv_states = conv_states.to(device=mixer.conv1d.weight.device)
    postconv = torch.sum(conv_states * mixer.conv1d.weight.squeeze(1), dim=-1)
    if mixer.use_conv_bias:
        postconv = postconv + mixer.conv1d.bias
    postconv = mixer.act(postconv)
    postconv = apply_mask_to_padding_states(postconv, attention_mask)
    record["POSTCONV_INPUT"] = postconv.detach().clone()
    hidden, B, C = torch.split(
        postconv,
        [mixer.intermediate_size, mixer.n_groups*mixer.ssm_state_size,
         mixer.n_groups*mixer.ssm_state_size], dim=-1)
    A = -torch.exp(mixer.A_log.float())
    cache_device = cache_params.layers[mixer.layer_idx].recurrent_states.device
    dt = dt[:, 0, :][:, None, ...]
    dt = dt.transpose(1, 2).expand(batch, dt.shape[-1], mixer.head_dim)
    dt_bias = mixer.dt_bias[..., None].expand(mixer.dt_bias.shape[0], mixer.head_dim)
    dt = F.softplus(dt + dt_bias.to(dt.dtype))
    dt = torch.clamp(dt, mixer.time_step_limit[0], mixer.time_step_limit[1])
    dt = take("DT", dt)
    A = A[..., None, None].expand(
        mixer.num_heads, mixer.head_dim, mixer.ssm_state_size).to(dtype=torch.float32)
    dA = torch.exp(dt[..., None] * A).to(device=cache_device)
    record["DA"] = dA.detach().clone()
    B = B.reshape(batch, mixer.n_groups, -1)[..., None, :]
    B = B.expand(batch, mixer.n_groups, mixer.num_heads//mixer.n_groups,
                 B.shape[-1]).contiguous().reshape(batch, -1, B.shape[-1])
    B = take("B", B)
    dB = dt[..., None] * B[..., None, :]
    hidden = hidden.reshape(batch, -1, mixer.head_dim)
    hidden = take("X", hidden)
    dBx = (dB * hidden[..., None]).to(device=cache_device)
    dBx = take("DBX", dBx)
    initial_state = cache_params.layers[mixer.layer_idx].recurrent_states
    initial_state = take("S", initial_state)
    state = initial_state*dA + dBx
    state = cache_params.update_recurrent_state(state, mixer.layer_idx)
    record["S_PRIME"] = state.detach().clone()
    C = C.reshape(batch, mixer.n_groups, -1)[..., None, :]
    C = C.expand(batch, mixer.n_groups, mixer.num_heads//mixer.n_groups,
                 C.shape[-1]).contiguous().reshape(batch, -1, C.shape[-1])
    C = take("C", C)
    state = state.to(device=C.device, dtype=C.dtype)
    reshaped_state = state.view(batch*mixer.num_heads, mixer.head_dim, mixer.ssm_state_size)
    reshaped_C = C.view(batch*mixer.num_heads, mixer.ssm_state_size, 1)
    y = torch.bmm(reshaped_state, reshaped_C).view(
        batch, mixer.num_heads, mixer.head_dim)
    record["READ_RAW"] = y.detach().clone()
    D = mixer.D[..., None].expand(mixer.D.shape[0], mixer.head_dim)
    y = (y + hidden*D).to(y.dtype)
    y = y.reshape(batch, -1)[:, None, ...]
    gate = take("OUTPUT_GATE", gate)
    scan = mixer.norm(y, gate) if mixer.mamba_rms_norm else y*F.silu(gate)
    record["NORMALIZED_OUTPUT"] = scan.detach().clone()
    result = mixer.out_proj(scan.to(dtype))
    record["MIXER_OUTPUT"] = result.detach().clone()
    return result


class FalconPrimitiveHook(AbstractContextManager):
    def __init__(self, model, layer: int, replacements: dict | None = None):
        self.model = model
        self.layer = int(layer)
        self.replacements = replacements or {}
        if any(name not in FACTOR_KEYS for name in self.replacements):
            raise ValueError("Unregistered primitive factor")
        self.capture = {}
        self._original = None
        self._module = None

    def __enter__(self):
        self._module = self.model.model.layers[self.layer].mamba
        self._original = self._module.torch_forward
        def forward(input_states, cache_params=None, attention_mask=None):
            return _falcon_step(self._module, input_states, cache_params,
                                attention_mask, self.replacements, self.capture)
        self._module.torch_forward = forward
        return self

    def __exit__(self, exc_type, exc, tb):
        self._module.torch_forward = self._original
        return False
