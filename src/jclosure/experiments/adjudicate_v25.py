"""Formal V25 outcome adjudication."""
from __future__ import annotations
import json
from pathlib import Path
from jclosure.protocol_v25 import stage_freeze,verify_stage
from jclosure.provenance import sha256_file,write_json_atomic
OUT=Path("results/v25/processed")
SOURCE="src/jclosure/experiments/adjudicate_v25.py"
def run(root:Path)->dict:
 verify_stage(root,"zero_rank_amendment");x=json.loads((root/OUT/"v25_integrated_analysis.json").read_text());convergence=json.loads((root/OUT/"zero_rank_amendment_v25.json").read_text())
 A=bool(x["interaction"]["STRONG_DISTRIBUTED_NONLINEAR_INTERACTION"]);B=bool(x["cancellation"]["CAUSAL_CANCELLATION_SUPPORTED"]);C=bool(convergence["MANY_TO_ONE_CAUSAL_CONVERGENCE"]);D=False;E=bool(x["readout"]["READOUT_COMPRESSION_SUPPORTED"] and not D);F=bool(x["background"]["BACKGROUND_CONDITIONED_CAUSAL_EFFECT"]);G=bool(x["CAUSAL_EFFECT_REALIZATION_DIMENSION_IDENTIFIED"]);H=bool(not any((A,B,C,F,G,E)))
 flags={"V25_A":A,"V25_B":B,"V25_C":C,"V25_D":D,"V25_E":E,"V25_F":F,"V25_G":G,"V25_H":H};names=[{"V25_A":"V25-A_STRONG_DISTRIBUTED_NONLINEAR_INTERACTION","V25_B":"V25-B_CAUSAL_CANCELLATION_SUPPORTED","V25_C":"V25-C_MANY_TO_ONE_CAUSAL_CONVERGENCE","V25_D":"V25-D_INTERNAL_CAUSAL_CONVERGENCE","V25_E":"V25-E_READOUT_COMPRESSION_ONLY","V25_F":"V25-F_BACKGROUND_CONDITIONED_RESPONSE_COORDINATES","V25_G":"V25-G_DISTRIBUTED_CAUSAL_REALIZATION_DIMENSION_IDENTIFIED","V25_H":"V25-H_NO_SIMPLE_DISTRIBUTED_DECOMPOSITION"}[k] for k,v in flags.items() if v]
 result={"formal_outcomes":names,"primary_outcome":"+".join(names),**flags,"LOW_RANK_RESPONSE_GEOMETRY_RECONFIRMED":x["rank_audit"]["LOW_RANK_RESPONSE_GEOMETRY_RECONFIRMED"],"V24_RANK1_STATUS":x["rank_audit"]["V24_RANK1_STATUS"],"corrected_convergence":convergence,"LOW_DIMENSIONAL_BEHAVIORAL_EFFECT_WITHOUT_LOW_DIMENSIONAL_CAUSAL_REALIZATION":bool((C or E) and not G),"causal_effect_realization_dimension":x["causal_effect_realization_dimension"],"independent_final_status":"FROZEN_AND_UNOPENED_VALIDATION_MECHANISMS_NOT_INDEPENDENTLY_FINALIZED","H2_REMAINS":True,"H3_AUTHORIZED":False,"DYNAMIC_STATE_SEARCH_AUTHORIZED":False,"historical_final_opened":False,"v25_independent_final_opened":False,"wording":"Response convergence and causal effect realization are not equated with compact model state."}
 output=root/OUT/"v25_adjudication.json";write_json_atomic(output,result)
 frozen=stage_freeze(root,"adjudication",[SOURCE,str(OUT/"v25_integrated_analysis.json"),str(OUT/"zero_rank_amendment_v25.json"),str(OUT/"v25_adjudication.json")],{"adjudication_sha256":sha256_file(output),"formal_outcomes":names,"historical_final_opened":False,"v25_independent_final_opened":False})
 return {**result,"freeze_digest":frozen["freeze_digest"]}
if __name__=="__main__":print(json.dumps(run(Path.cwd()),indent=2))
