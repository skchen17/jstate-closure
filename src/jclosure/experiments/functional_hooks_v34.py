"""Exact one-token native functional-stage capture and bidirectional replacement.

Only future one-token cached decoding is instrumented. The Qwen kernel is the
installed torch recurrence reproduced operation-for-operation; the Falcon
single-token Mamba path is similarly reproduced. Calibration replay must pass
bitwise before any formal mediation run is allowed.
"""
from __future__ import annotations

from contextlib import AbstractContextManager

import torch
import torch.nn.functional as F

from jclosure.experiments.transaction_v28 import thash


STAGES=("TRUE_UPDATE","TRANSFORMED_CONTROL","POSTCONV_INPUT","RECURRENT_READ","RESIDUAL_INTEGRATION")


def _qwen_step(query,key,value,g,beta,initial_state,output_final_state,use_qk_l2norm_in_kernel,write_override=None):
    from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm
    dtype=query.dtype
    if use_qk_l2norm_in_kernel:
        query=l2norm(query,dim=-1,eps=1e-6)
        key=l2norm(key,dim=-1,eps=1e-6)
    query,key,value,beta,g=[x.transpose(1,2).contiguous().to(torch.float32) for x in (query,key,value,beta,g)]
    if query.shape[2]!=1:raise RuntimeError("V34 Qwen kernel instrument only supports one token")
    batch,heads,_,key_dim=key.shape
    value_dim=value.shape[-1]
    query=query*(1/(query.shape[-1]**0.5))
    output=torch.zeros(batch,heads,1,value_dim,dtype=value.dtype,device=value.device)
    state=torch.zeros(batch,heads,key_dim,value_dim,dtype=value.dtype,device=value.device) if initial_state is None else initial_state.to(value)
    q_t,k_t,v_t=query[:,:,0],key[:,:,0],value[:,:,0]
    g_t=g[:,:,0].exp().unsqueeze(-1).unsqueeze(-1)
    beta_t=beta[:,:,0].unsqueeze(-1)
    decayed=state*g_t
    kv_mem=(decayed*k_t.unsqueeze(-1)).sum(dim=-2)
    delta=(v_t-kv_mem)*beta_t
    natural_write=k_t.unsqueeze(-1)*delta.unsqueeze(-2)
    write=natural_write if write_override is None else write_override
    if write.shape!=natural_write.shape or write.dtype!=natural_write.dtype:raise RuntimeError("V34 Qwen true update shape/dtype mismatch")
    state=decayed+write
    output[:,:,0]=(state*q_t.unsqueeze(-1)).sum(dim=-2)
    return output.transpose(1,2).contiguous().to(dtype),(state if output_final_state else None),natural_write


def _falcon_step(mixer,input_states,cache_params,attention_mask,stage,reference,record):
    from transformers.models.falcon_h1.modeling_falcon_h1 import apply_mask_to_padding_states
    batch,seq_len,_=input_states.shape
    if seq_len!=1 or cache_params is None or not cache_params.has_previous_state(mixer.layer_idx):raise RuntimeError("V34 Falcon instrument only supports cached one-token decode")
    dtype=input_states.dtype
    input_states=apply_mask_to_padding_states(input_states,attention_mask)
    input_states=input_states*mixer.ssm_in_multiplier
    projected=mixer.in_proj(input_states)*mixer.mup_vector
    gate,conv_input,dt=projected.split([mixer.intermediate_size,mixer.conv_dim,mixer.num_heads],dim=-1)
    conv_input=conv_input.transpose(1,2)
    conv_states=cache_params.update_conv_state(conv_input,mixer.layer_idx)
    conv_states=conv_states.to(device=mixer.conv1d.weight.device)
    postconv=torch.sum(conv_states*mixer.conv1d.weight.squeeze(1),dim=-1)
    if mixer.use_conv_bias:postconv=postconv+mixer.conv1d.bias
    postconv=mixer.act(postconv)
    postconv=apply_mask_to_padding_states(postconv,attention_mask)
    record["POSTCONV_INPUT"]=postconv.detach().clone()
    if stage=="POSTCONV_INPUT":
        postconv=_replacement(reference,mixer.layer_idx,"POSTCONV_INPUT",postconv,record)
    hidden,B,C=torch.split(postconv,[mixer.intermediate_size,mixer.n_groups*mixer.ssm_state_size,mixer.n_groups*mixer.ssm_state_size],dim=-1)
    A=-torch.exp(mixer.A_log.float())
    cache_device=cache_params.layers[mixer.layer_idx].recurrent_states.device
    dt=dt[:,0,:][:,None,...]
    dt=dt.transpose(1,2).expand(batch,dt.shape[-1],mixer.head_dim)
    dt_bias=mixer.dt_bias[...,None].expand(mixer.dt_bias.shape[0],mixer.head_dim)
    dt=F.softplus(dt+dt_bias.to(dt.dtype))
    dt=torch.clamp(dt,mixer.time_step_limit[0],mixer.time_step_limit[1])
    record["TRANSFORMED_CONTROL"]=dt.detach().clone()
    if stage=="TRANSFORMED_CONTROL":
        dt=_replacement(reference,mixer.layer_idx,"TRANSFORMED_CONTROL",dt,record)
    A=A[...,None,None].expand(mixer.num_heads,mixer.head_dim,mixer.ssm_state_size).to(dtype=torch.float32)
    dA=torch.exp(dt[...,None]*A).to(device=cache_device)
    B=B.reshape(batch,mixer.n_groups,-1)[...,None,:]
    B=B.expand(batch,mixer.n_groups,mixer.num_heads//mixer.n_groups,B.shape[-1]).contiguous().reshape(batch,-1,B.shape[-1])
    dB=dt[...,None]*B[...,None,:]
    hidden=hidden.reshape(batch,-1,mixer.head_dim)
    dBx=(dB*hidden[...,None]).to(device=cache_device)
    record["TRUE_UPDATE"]=dBx.detach().clone()
    if stage=="TRUE_UPDATE":
        dBx=_replacement(reference,mixer.layer_idx,"TRUE_UPDATE",dBx,record)
    state=cache_params.layers[mixer.layer_idx].recurrent_states*dA+dBx
    state=cache_params.update_recurrent_state(state,mixer.layer_idx)
    C=C.reshape(batch,mixer.n_groups,-1)[...,None,:]
    C=C.expand(batch,mixer.n_groups,mixer.num_heads//mixer.n_groups,C.shape[-1]).contiguous().reshape(batch,-1,C.shape[-1])
    state=state.to(device=C.device,dtype=C.dtype)
    reshaped_state=state.view(batch*mixer.num_heads,mixer.head_dim,mixer.ssm_state_size)
    reshaped_C=C.view(batch*mixer.num_heads,mixer.ssm_state_size,1)
    y=torch.bmm(reshaped_state,reshaped_C).view(batch,mixer.num_heads,mixer.head_dim)
    D=mixer.D[...,None].expand(mixer.D.shape[0],mixer.head_dim)
    y=(y+hidden*D).to(y.dtype)
    y=y.reshape(batch,-1)[:,None,...]
    scan=mixer.norm(y,gate) if mixer.mamba_rms_norm else y*F.silu(gate)
    result=mixer.out_proj(scan.to(dtype))
    return result


def _replacement(reference,layer,stage,current,record):
    if reference is None or layer not in reference or stage not in reference[layer]:raise RuntimeError(f"V34 missing reference {layer}:{stage}")
    requested=reference[layer][stage]
    if requested.shape!=current.shape or requested.dtype!=current.dtype or requested.device!=current.device:
        raise RuntimeError(f"V34 {stage} replacement topology mismatch layer {layer}")
    realized=requested.detach().clone()
    if not torch.equal(realized,requested):raise RuntimeError("V34 realized write mismatch")
    record["requested_hash"]=thash(requested)
    record["realized_hash"]=thash(realized)
    record["native_hash"]=thash(current)
    record["exact_writeback"]=True
    return realized


class FunctionalIntervention(AbstractContextManager):
    def __init__(self,model,key,stage=None,reference=None):
        if stage is not None and stage not in STAGES:raise ValueError(stage)
        self.model=model;self.key=key;self.stage=stage;self.reference=reference
        self.capture={};self._original=[];self._handles=[]

    def __enter__(self):
        spec_layers=[(i,self.model.model.layers[i]) for i in range(len(self.model.model.layers)) if self.key=="F" or self.model.model.layers[i].layer_type=="linear_attention"]
        for layer,block in spec_layers:
            module=block.linear_attn if self.key=="Q" else block.mamba
            record={};self.capture[layer]=record
            if self.key=="Q":
                original=module.recurrent_gated_delta_rule
                def kernel(query,key,value,*,g,beta,initial_state,output_final_state,use_qk_l2norm_in_kernel,layer=layer,record=record):
                    record["POSTCONV_INPUT"]=(query.detach().clone(),key.detach().clone(),value.detach().clone())
                    record["TRANSFORMED_CONTROL"]=(g.detach().clone(),beta.detach().clone())
                    if self.stage=="POSTCONV_INPUT":
                        values=self.reference[layer]["POSTCONV_INPUT"]
                        query,key,value=tuple(_replacement_component(values[i],x,record,f"POSTCONV_INPUT_{i}") for i,x in enumerate((query,key,value)))
                        record["exact_writeback"]=True
                    if self.stage=="TRANSFORMED_CONTROL":
                        values=self.reference[layer]["TRANSFORMED_CONTROL"]
                        g,beta=tuple(_replacement_component(values[i],x,record,f"TRANSFORMED_CONTROL_{i}") for i,x in enumerate((g,beta)))
                        record["exact_writeback"]=True
                    first,second,natural=_qwen_step(query,key,value,g,beta,initial_state,output_final_state,use_qk_l2norm_in_kernel)
                    record["TRUE_UPDATE"]=natural.detach().clone()
                    if self.stage=="TRUE_UPDATE":
                        replacement=_replacement(self.reference,layer,"TRUE_UPDATE",natural,record)
                        first,second,_=_qwen_step(query,key,value,g,beta,initial_state,output_final_state,use_qk_l2norm_in_kernel,replacement)
                    record["RECURRENT_READ"]=first.detach().clone()
                    return first,second
                module.recurrent_gated_delta_rule=kernel
                self._original.append((module,"recurrent_gated_delta_rule",original))
            else:
                original=module.torch_forward
                def falcon(input_states,cache_params=None,attention_mask=None,layer=layer,record=record,module=module):
                    return _falcon_step(module,input_states,cache_params,attention_mask,self.stage,self.reference,record)
                module.torch_forward=falcon
                self._original.append((module,"torch_forward",original))
            def output_hook(_m,_inp,out,layer=layer,record=record):
                record["RECURRENT_READ"]=out.detach().clone()
                record["RESIDUAL_INTEGRATION"]=out.detach().clone()
                if self.stage in ("RECURRENT_READ","RESIDUAL_INTEGRATION"):
                    return _replacement(self.reference,layer,self.stage,out,record)
                return None
            self._handles.append(module.register_forward_hook(output_hook))
        return self

    def __exit__(self,exc_type,exc,tb):
        for handle in self._handles:handle.remove()
        for module,name,original in self._original:setattr(module,name,original)
        self._handles.clear();self._original.clear()
        return False


def _replacement_component(reference,current,record,name):
    if reference.shape!=current.shape or reference.dtype!=current.dtype or reference.device!=current.device:raise RuntimeError(f"V34 {name} topology mismatch")
    value=reference.detach().clone()
    if not torch.equal(value,reference):raise RuntimeError("V34 component write mismatch")
    record[name+"_requested_hash"]=thash(reference)
    record[name+"_realized_hash"]=thash(value)
    record[name+"_native_hash"]=thash(current)
    return value
