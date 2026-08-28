"""Pinned model/lens loading with artifact verification."""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from jclosure.provenance import verify_sha256


@dataclass(frozen=True)
class ModelBundle:
    hf_model: Any
    tokenizer: Any
    lens_model: Any
    lens: Any
    model_id: str
    model_revision: str
    lens_path: Path
    lens_revision: str

    @property
    def layers(self):
        return self.lens_model.layers

    @property
    def unembedding_weight(self) -> torch.Tensor:
        return self.lens_model._lm_head.weight  # reference adapter owns this path

    @torch.no_grad()
    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        output = self.hf_model(input_ids=input_ids, use_cache=False)
        return output.logits


def _artifact_root() -> Path:
    configured = os.environ.get("JCLOSURE_ARTIFACT_DIR")
    root = Path(configured) if configured else Path.home() / ".cache" / "jclosure"
    root.mkdir(parents=True, exist_ok=True)
    return root


def download_lens(
    *,
    repo_id: str,
    revision: str,
    filename: str,
    sha256: str,
    local_path: str | Path | None = None,
) -> Path:
    """Download through huggingface_hub, with a verified direct fallback."""

    if local_path is not None:
        path = Path(local_path).expanduser().resolve()
        verify_sha256(path, sha256)
        return path
    target = _artifact_root() / "lenses" / revision / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        verify_sha256(target, sha256)
        return target
    try:
        from huggingface_hub import hf_hub_download

        downloaded = Path(
            hf_hub_download(repo_id=repo_id, revision=revision, filename=filename)
        )
        verify_sha256(downloaded, sha256)
        return downloaded
    except Exception as hub_error:
        url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
        partial = target.with_suffix(target.suffix + ".partial")
        try:
            with urllib.request.urlopen(url) as response, partial.open("wb") as output:
                while chunk := response.read(8 * 1024 * 1024):
                    output.write(chunk)
            os.replace(partial, target)
            verify_sha256(target, sha256)
            return target
        except Exception as direct_error:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"lens download failed via Hub ({hub_error}) and direct URL ({direct_error})"
            ) from direct_error


def load_model_bundle(config: dict[str, Any]) -> ModelBundle:
    """Load the exact configured model and lens; never silently change revisions."""

    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = config["model"]
    lens_cfg = config["lens"]
    dtype_name = str(model_cfg.get("dtype", "bfloat16"))
    dtype = getattr(torch, dtype_name)
    load_kwargs: dict[str, Any] = {
        "revision": str(model_cfg["revision"]),
        "torch_dtype": dtype,
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
    }
    device_map = model_cfg.get("device_map", "cuda")
    if device_map == "cuda":
        load_kwargs["device_map"] = {"": int(model_cfg.get("device", 0))}
    else:
        load_kwargs["device_map"] = device_map
        if "max_memory" in model_cfg:
            load_kwargs["max_memory"] = model_cfg["max_memory"]
        if "offload_folder" in model_cfg:
            load_kwargs["offload_folder"] = model_cfg["offload_folder"]

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_cfg["id"]),
        revision=str(model_cfg["revision"]),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    try:
        hf_model = AutoModelForCausalLM.from_pretrained(
            str(model_cfg["id"]), **load_kwargs
        )
    except ValueError as causal_error:
        try:
            from transformers import AutoModelForImageTextToText

            hf_model = AutoModelForImageTextToText.from_pretrained(
                str(model_cfg["id"]), **load_kwargs
            )
        except Exception as multimodal_error:
            raise causal_error from multimodal_error
    hf_model.eval()
    lens_model = jlens.from_hf(
        hf_model,
        tokenizer,
        compile=bool(model_cfg.get("compile", False)),
        force_bos=bool(model_cfg.get("force_bos", True)),
    )
    lens_path = download_lens(
        repo_id=str(lens_cfg["repo"]),
        revision=str(lens_cfg["revision"]),
        filename=str(lens_cfg["file"]),
        sha256=str(lens_cfg["sha256"]),
        local_path=lens_cfg.get("local_path"),
    )
    lens = jlens.JacobianLens.load(str(lens_path))
    if lens.d_model != lens_model.d_model:
        raise ValueError(
            f"lens d_model={lens.d_model} does not match model d_model={lens_model.d_model}"
        )
    invalid_layers = [
        layer for layer in lens.source_layers if not 0 <= layer < lens_model.n_layers
    ]
    if invalid_layers:
        raise ValueError(f"lens contains invalid model layers: {invalid_layers}")
    return ModelBundle(
        hf_model=hf_model,
        tokenizer=tokenizer,
        lens_model=lens_model,
        lens=lens,
        model_id=str(model_cfg["id"]),
        model_revision=str(model_cfg["revision"]),
        lens_path=lens_path,
        lens_revision=str(lens_cfg["revision"]),
    )
