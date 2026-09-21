"""Append-only correction to V21 orthogonal Procrustes nuclear-norm algebra."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jclosure.experiments import geometry_analysis_v21 as geometry
from jclosure.protocol_v21 import stage_freeze, verify_stage

SOURCE="src/jclosure/experiments/geometry_analysis_amend_v21.py"


def corrected_decompose(a:np.ndarray,b:np.ndarray,sa:dict,sb:dict)->dict:
    norm_a=max(float(np.linalg.norm(a)),1e-20)
    norm_b=max(float(np.linalg.norm(b)),1e-20)
    pre=float(np.linalg.norm(b-a)/norm_b)
    # Nonzero singular values of B A^T equal those of
    # Sigma_B (V_B^T V_A) Sigma_A; A^T B is not equivalent.
    core=(sb["singular"][:,None]*(sb["vh"]@sa["vh"].T))*sa["singular"][None,:]
    nuclear=float(np.sum(np.linalg.svd(core,compute_uv=False)))
    post=float(np.sqrt(max(norm_a**2+norm_b**2-2*nuclear,0.0))/norm_b)
    gain=nuclear/max(norm_a**2,1e-20)
    post_gain=float(np.sqrt(max(norm_b**2-nuclear**2/max(norm_a**2,1e-20),0.0))/norm_b)
    return {"pre_alignment_relative_error":pre,
            "post_orthogonal_Procrustes_relative_error":post,
            "post_orthogonal_and_gain_relative_error":post_gain,
            "optimal_Procrustes_gain":gain,
            "frobenius_gain_ratio":norm_b/norm_a,
            "singular_spectrum_JS_divergence":geometry._spectral_js(sa,sb),
            "r95_difference":sb["r95"]-sa["r95"]}


def prepare(root:Path)->dict:
    verify_stage(root,"paired_geometry_analysis_design")
    return stage_freeze(root,"paired_geometry_Procrustes_amendment_1",
                        [SOURCE,"artifacts/action_coordinate_geometry_v21_paired_geometry_analysis_design.freeze.json",
                         "tests/test_v21.py"],
                        {"reason":"CPU exact-rotation invariant caught incorrect use of nuclear_norm(A^T B); correct output-space orthogonal Procrustes uses nuclear_norm(B A^T) via compact SVD core.",
                         "original_analysis_rows_observed":0,
                         "raw_JVP_finite_matrices_unchanged":True,
                         "final_six_action_responses_opened":False})


def run(root:Path)->dict:
    verify_stage(root,"paired_geometry_Procrustes_amendment_1")
    old=geometry._decompose
    geometry._decompose=corrected_decompose
    try:
        return geometry.run(root)
    finally:
        geometry._decompose=old


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("prepare","run"))
    args=parser.parse_args()
    print(json.dumps(prepare(Path.cwd()) if args.stage=="prepare" else run(Path.cwd()),indent=2))
