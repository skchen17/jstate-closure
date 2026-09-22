"""Generate all V30 scientific reports, one-file bundle and integrity index."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments.global_basis_v30 import SCRATCH
from jclosure.protocol_v30 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/report_v30.py"
OUT = Path("results/v30/processed")
ORDER = (
    "TOKEN_PAIR_LIBRARY", "LOCAL_WRITE_GEOMETRY", "GLOBAL_WRITE_BASIS", "STATE_OOD", "TOKEN_OOD", "JOINT_OOD", "FAMILY_OOD", "GLOBAL_VS_LOCAL", "CROSS_STATE_COORDINATE_TRANSPORT", "STATE_DEPENDENT_WRITE_ATLAS", "MULTI_PROBE_WRITE_SIGNATURE", "WRITE_HORIZON", "CONV_LAYER_PROFILE", "MINIMAL_CONV_CARRIER", "REC_CONV_CONDITIONAL_EFFECT", "WRITE_RANK_SCALING", "NATURAL_VS_ARTIFICIAL_COORDINATES", "STRICT_WRITE_INTERFACE_AUDIT", "EXECUTION_MANIFEST", "SCIENTIFIC_ANSWERS", "COMPLETE_REPORT",
)
ROLES = ("development", "validation")


def f(x):
    return "n/a" if x is None or pd.isna(x) else f"{float(x):.3f}"


def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n|" + "|".join("---" for _ in headers) + "|\n" + "\n".join("| " + " | ".join(str(v).replace("|", "\\|").replace("\n", "\\n") for v in row) + " |" for row in rows) + "\n"


def metrics(frame, conditions, axis=None):
    x = frame if axis is None or "axis" not in frame else frame[frame.axis == axis]
    rows = []
    for condition in conditions:
        z = x[x.condition == condition]
        rows.append((condition, len(z), f(z.donor_cosine.median()) if len(z) else "n/a", f(z.magnitude_ratio.median()) if len(z) else "n/a", f(z.relative_l2_to_donor.median()) if len(z) else "n/a"))
    return table(("condition", "n", "cosine", "magnitude", "relative L2"), rows)


def h(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root: Path):
    verify_stage(root, "adjudication")
    report_dir = root / "reports"
    d = json.loads((root / OUT / "design_v30.json").read_text())
    p = json.loads((root / OUT / "execution_plan_v30.json").read_text())
    fit = json.loads((root / OUT / "global_basis_fit_v30.json").read_text())
    final = json.loads((root / OUT / "final_opening_v30.json").read_text())
    adj = json.loads((root / OUT / "v30_adjudication.json").read_text())
    amendment = json.loads((root / OUT / "transport_source_amendment_v30.json").read_text())
    g = {r: pd.read_parquet(root / OUT / f"causal_global_{r}_v30.parquet") for r in ROLES}
    local = {r: pd.read_parquet(root / OUT / f"local_transport_{r}_v30.parquet") for r in ROLES}
    geometry = {r: pd.read_parquet(root / OUT / f"local_geometry_{r}_v30.parquet") for r in ROLES}
    local_basis = {r: json.loads((root / OUT / f"local_basis_records_{r}_v30.json").read_text())["records"] for r in ROLES}
    conv = {r: pd.read_parquet(root / OUT / f"conv_layer_profile_{r}_v30.parquet") for r in ROLES}
    conditional = {r: pd.read_parquet(root / OUT / f"rec_conv_conditional_{r}_v30.parquet") for r in ROLES}
    controls = {r: pd.read_parquet(root / OUT / f"coordinate_controls_{r}_v30.parquet") for r in ROLES}
    horizon = {r: pd.read_parquet(root / OUT / f"write_horizon_{r}_v30.parquet") for r in ROLES}
    conv_selection = json.loads((root / OUT / "conv_minimal_selection_v30.json").read_text())
    rank = json.loads((root / OUT / "write_rank_scaling_v30.json").read_text())
    rank120 = json.loads((root / OUT / "write_rank_scaling_120_v30.json").read_text())
    gate = p["gates"]
    reports = {}
    def add(name, title, body):
        reports[name] = f"# {title} — V30\n\n{body.strip()}\n"

    library_rows = [(x["role"], x["pair_id"], x["anchor_token_id"], x["candidate_token_id"], json.dumps(x["candidate_surface"], ensure_ascii=False), f(x["calibration_mean_rank"])) for x in d["token_pair_library"]]
    add("TOKEN_PAIR_LIBRARY", "Token Pair Library", f"Frozen calibration-only response-blind library: 88 anchor→candidate contrasts, split 72 TOKEN_TRAIN / 8 TOKEN_VALIDATION / 8 TOKEN_FINAL. Anchor token ID 25 (`:`); candidates are tokenizer tokens common at rank ≤1024 across 25 calibration prefixes. Every token's state eligibility is rank ≤1024, not post-write success. The 88 pairs exceed the suggested 24–48 because centered local k64 needs at least 65 TRAIN contrasts plus disjoint token holdouts. This library is dominated by frequent punctuation, whitespace and lexical fragments; it is **not** a semantic-command panel. No claim about arbitrary language tokens follows. The 50 independent-final states and 8 TOKEN_FINAL pairs stayed sealed.\n\n{table(('split','pair ID','anchor','candidate','surface','calibration mean rank'),library_rows)}\nMachine record: `results/v30/processed/design_v30.json`; library hash `{d['library_hash']}`.")

    local_rank_rows = [(role, len(local_basis[role]), f(np.median([x["ranks"]["r90"] for x in local_basis[role]])), f(np.median([x["ranks"]["r95"] for x in local_basis[role]])), f(np.median([x["ranks"]["r99"] for x in local_basis[role]])), f(geometry[role].principal_cosine_median.median()), f(geometry[role].principal_angle_max_degrees.median()), f(geometry[role].chordal_distance.median())) for role in ROLES]
    add("LOCAL_WRITE_GEOMETRY", "Local Write Geometry", f"For each target, a centered local REC+Conv basis was fit only from its 70–72 eligible TOKEN_TRAIN contrasts. k=8/16/32/64 was tested; held-out token writes were not used to fit it. The 32D source–target principal-angle audit is descriptive, not a causal language proof.\n\n{table(('role','states','median r90','median r95','median r99','median 32D principal cosine','median max angle °','median chordal'),local_rank_rows)}\nThe local TRAIN spectrum can have r95 ≈54 while the unseen-token k64 natural-write reconstruction median relative L2 remains {f(local['development'][local['development'].condition=='LOCAL_ORACLE_k64'].write_relative_l2.median())} development and {f(local['validation'][local['validation'].condition=='LOCAL_ORACLE_k64'].write_relative_l2.median())} validation. Thus TRAIN geometry does not establish unseen-token sufficiency.")

    fit_rows = [(name, z["training_rows"], z["rank"], z["ranks"]["r90"], z["ranks"]["r95"], z["ranks"]["r99"], z["basis_sha256"][:16]) for name, z in fit["models"].items()]
    add("GLOBAL_WRITE_BASIS", "Global Write Basis", f"The fixed 13,369,344-component REC+Conv write vector was fit on 90 development states × 8 frozen TOKEN_TRAIN contrasts = 720 rows. These are centered **affine** PCA approximations (mean + Uc), not exact write dimensions or model-state dimensions. All basis tensors remain off-repository scratch; their SHA-256 hashes are committed.\n\n{table(('basis','TRAIN rows','rank','r90','r95','r99','basis hash prefix'),fit_rows)}\nThe global r95=131 already exceeds k64. Geometry alone is not the verdict: the held-out causal OOD results below decide fidelity.")

    state_tables = "\n".join(f"### {role}\n\n" + metrics(g[role], ("EXACT_REC_CONV", "EXACT_CONV", "EXACT_KV", "GLOBAL_k32", "GLOBAL_k64", "FAMILY_k32", "FAMILY_k64", "LOFO_k32", "LOFO_k64"), "STATE_OOD") for role in ROLES)
    add("STATE_OOD", "Unseen-State Causal Fidelity", f"STATE-OOD uses unseen development/validation states with TOKEN_TRAIN pairs that were eligible in the source library. The future endpoint is the concatenated normalized signature from four probes selected before writes. Strong gate: cosine ≥{gate['cosine_min']}, magnitude [{gate['magnitude_min']},{gate['magnitude_max']}], relative L2 ≤{gate['relative_l2_max']}, success fraction ≥{gate['success_fraction_min']}, ≥{gate['families_required']}/5 families.\n\n{state_tables}\nThe fixed global k64 basis approaches but does not pass the L2 gate on state OOD; exact REC+Conv remains the native causal ceiling. Family-fixed and LOFO rows are separate controls, not replacements for one global basis.")

    add("TOKEN_OOD", "Unseen-Token Causal Fidelity", "TOKEN-OOD tests 10 seen training states × 2 TOKEN_VALIDATION pairs each; no held-out token pair entered basis fitting.\n\n" + metrics(g["development"], ("EXACT_REC_CONV", "GLOBAL_k32", "GLOBAL_k64", "FAMILY_k64", "LOFO_k64"), "TOKEN_OOD") + "\nEven on seen states, k64 fails the causal gate on new token contrasts. Independent validation of unseen state plus unseen token is reported under JOINT-OOD. TOKEN_FINAL remains unopened.")

    joint_tables = "\n".join(f"### {role}\n\n" + metrics(g[role], ("EXACT_REC_CONV", "EXACT_CONV", "EXACT_KV", "GLOBAL_k32", "GLOBAL_k64", "FAMILY_k64", "LOFO_k64"), "JOINT_OOD") for role in ROLES)
    add("JOINT_OOD", "Joint State-and-Token OOD", f"Primary difficult test: both target state and token contrast are outside the global basis fit. Each role uses 2 prospectively frozen TOKEN_VALIDATION contrasts per state.\n\n{joint_tables}\nGlobal k64 has validation median cosine {f(g['validation'][(g['validation'].condition=='GLOBAL_k64')&(g['validation'].axis=='JOINT_OOD')].donor_cosine.median())}, relative L2 {f(g['validation'][(g['validation'].condition=='GLOBAL_k64')&(g['validation'].axis=='JOINT_OOD')].relative_l2_to_donor.median())}; exact REC+Conv is {f(g['validation'][(g['validation'].condition=='EXACT_REC_CONV')&(g['validation'].axis=='JOINT_OOD')].relative_l2_to_donor.median())} L2. A fixed compact global basis is not causally sufficient here.")

    fam_rows = []
    for role in ROLES:
        gg = g[role][g[role].axis == "JOINT_OOD"]
        for fam in d["families"]:
            z = gg[gg.family == fam]
            fam_rows.append((role, fam, f(z[z.condition=='EXACT_REC_CONV'].relative_l2_to_donor.median()), f(z[z.condition=='EXACT_CONV'].relative_l2_to_donor.median()), f(z[z.condition=='GLOBAL_k32'].relative_l2_to_donor.median()), f(z[z.condition=='GLOBAL_k64'].relative_l2_to_donor.median()), f(z[z.condition=='LOFO_k64'].relative_l2_to_donor.median()), f(local[role][(local[role].family==fam)&(local[role].condition=='LOCAL_ORACLE_k64')].relative_l2_to_donor.median()), f(local[role][(local[role].family==fam)&(local[role].condition=='TRANSPORTED_LOCAL_RIDGE_k64')].relative_l2_to_donor.median())))
    add("FAMILY_OOD", "Leave-One-Family-Out Generalization", f"For each family, the LOFO basis excludes **all** TRAIN rows of that family. The table reports JOINT-OOD relative L2 separately; no failed family is averaged away.\n\n{table(('role','held-out family','exact REC+Conv','Conv','global32','global64','LOFO64','local oracle64','transport ridge64'),fam_rows)}\nNo k≤64 family-OOD causal representation meets the frozen ≥4/5-family rule in both development and validation. The exact natural ceiling remains distinct from compact generalization.")

    gl_rows = []
    for role in ROLES:
        for condition, frame in (("GLOBAL_k32",g[role][g[role].axis=='JOINT_OOD']), ("GLOBAL_k64",g[role][g[role].axis=='JOINT_OOD']), ("FAMILY_k64",g[role][g[role].axis=='JOINT_OOD']), ("LOCAL_ORACLE_k32",local[role]), ("LOCAL_ORACLE_k64",local[role]), ("TRANSPORTED_LOCAL_RIDGE_k32",local[role]), ("TRANSPORTED_LOCAL_RIDGE_k64",local[role]), ("TRANSPORTED_LOCAL_PROCRUSTES_k64",local[role])):
            z = frame[frame.condition==condition]
            gl_rows.append((role,condition,len(z),f(z.donor_cosine.median()),f(z.magnitude_ratio.median()),f(z.relative_l2_to_donor.median()),f(z.write_relative_l2.median())))
    add("GLOBAL_VS_LOCAL", "Global versus Local Causal Models", f"M0 fixed global, M1 family-fixed, M2 target local oracle, M3 source→target local transport are compared on **held-out target-state + held-out-token** cases. All local bases and transport maps use TOKEN_TRAIN only.\n\n{table(('role','model','n','cosine','magnitude','causal L2','write L2'),gl_rows)}\nLocal k64 and transported k64 are close, but both miss the strong causal ceiling. Consequently M0 is insufficient and M3 does not establish a shared command; M2 failure prevents attributing this solely to a bad transport map.")

    sources = json.loads((root / OUT / "local_source_bases_v30.json").read_text())["sources"]
    source_rows = [(fam, r["state_id"],len(r["train_pair_ids"]),r["ranks"]["r95"],r["basis_sha256"][:16]) for fam,r in sources.items()]
    add("CROSS_STATE_COORDINATE_TRANSPORT", "Cross-State Coordinate Transport", f"Five canonical source states were selected using only prewrite eligibility before V30 causal responses; this resource amendment changed only the source map and preserved all 280 frozen target–pair keys. Each source/target local map used ≥65 common TOKEN_TRAIN pairs, ridge α=1 or orthogonal Procrustes; target held-out token write never fit the map. The source's held-out contrast was projected into its TRAIN basis, mapped to the target's TRAIN basis, written into target-native REC+Conv cache, and judged against the target-native donor future. Raw source delta is **not** the primary test.\n\n{table(('family','canonical source','TRAIN pairs','source r95','basis hash prefix'),source_rows)}\n{metrics(local['development'],('LOCAL_ORACLE_k64','TRANSPORTED_LOCAL_RIDGE_k64','TRANSPORTED_LOCAL_PROCRUSTES_k64'))}\n{metrics(local['validation'],('LOCAL_ORACLE_k64','TRANSPORTED_LOCAL_RIDGE_k64','TRANSPORTED_LOCAL_PROCRUSTES_k64'))}\nNeither map passes unseen-token causal gates. Similarity to the failing local ceiling is not shared-language confirmation. Transport hashes are per-row in `local_transport_*_v30.parquet`.")

    add("STATE_DEPENDENT_WRITE_ATLAS", "State-Dependent Write Atlas", f"The 32D local principal-cosine median is {f(geometry['development'].principal_cosine_median.median())} development and {f(geometry['validation'].principal_cosine_median.median())} validation. Some directions rotate (development median maximum principal angle {f(geometry['development'].principal_angle_max_degrees.median())}°), but principal angles alone are descriptive. The defining atlas criterion requires target local k≤64 causal sufficiency on held-out tokens **and** train-only transport recovery after a weaker fixed global basis. The local oracle fails (validation k64 causal L2 {f(local['validation'][local['validation'].condition=='LOCAL_ORACLE_k64'].relative_l2_to_donor.median())}); therefore V30-C is not established. V29 same-background compression remains valid only for its tested regime. A fourth possibility—token-dependent write directions exceeding the tested compact span—remains open.")

    add("MULTI_PROBE_WRITE_SIGNATURE", "Multi-Probe Write Signature", f"All main V30 causal rows concatenate J, selected logits, semantic log-probabilities, workspace, broad vocabulary and late residual across **four frozen next-token probes**. This is stricter than a single next token. Validation JOINT-OOD: exact REC+Conv cosine {f(g['validation'][(g['validation'].axis=='JOINT_OOD')&(g['validation'].condition=='EXACT_REC_CONV')].donor_cosine.median())}, global k64 {f(g['validation'][(g['validation'].axis=='JOINT_OOD')&(g['validation'].condition=='GLOBAL_k64')].donor_cosine.median())}, local oracle k64 {f(local['validation'][local['validation'].condition=='LOCAL_ORACLE_k64'].donor_cosine.median())}. The four probes were fixed from prefix logits before current-token natural writes. This report does not claim semantic-coordinate labels.")

    horizon_rows = []
    for role in ROLES:
        for hh in (1,2,4):
            z = horizon[role][horizon[role].horizon==hh]
            for condition in ("EXACT_REC_CONV","EXACT_CONV","GLOBAL_k32","GLOBAL_k64"):
                y = z[z.condition==condition]
                horizon_rows.append((role,hh,condition,f(y.donor_cosine.median()),f(y.magnitude_ratio.median()),f(y.relative_l2_to_donor.median())))
    add("WRITE_HORIZON", "Write Horizon", f"A separate 10-state/role panel replays the same four prewrite-frozen tokens sequentially and compares h1/h2/h4; this is a single teacher-forced trajectory, not the four-probe concatenated primary endpoint.\n\n{table(('role','h','condition','cosine','magnitude','relative L2'),horizon_rows)}\nExact REC+Conv retains a donor-directed effect through h4 but development h4 does not meet the 0.30 L2 gate. Global compact writes do not regain qualification at later horizons. No transported-local h2/h4 claim is made because its h1 OOD gate failed.")

    single_rows=[]
    for j,layer in enumerate(p["conv_layers"]):
        row=[layer]
        for role in ROLES:
            z=conv[role][conv[role].condition==f"SINGLE_{j:02d}"]
            row += [f(z.donor_cosine.median()),f(z.relative_l2_to_donor.median())]
        single_rows.append(tuple(row))
    add("CONV_LAYER_PROFILE", "Conv Layer Profile", f"Exact native donor Conv fields were tested at every one of 24 recurrent layers on a prospectively frozen 10-state/role panel. All untouched cache fields are checked bitwise by the experiment. Full Conv: development cosine {f(conv['development'][conv['development'].condition=='FULL_CONV'].donor_cosine.median())}, L2 {f(conv['development'][conv['development'].condition=='FULL_CONV'].relative_l2_to_donor.median())}; validation cosine {f(conv['validation'][conv['validation'].condition=='FULL_CONV'].donor_cosine.median())}, L2 {f(conv['validation'][conv['validation'].condition=='FULL_CONV'].relative_l2_to_donor.median())}.\n\n{table(('layer','dev cos','dev L2','val cos','val L2'),single_rows)}\nQuartiles, halves, prefixes, suffixes, leave-quartile-out and 24 leave-single-out rows are preserved in `conv_layer_profile_*_v30.parquet`; a single layer's effect is not additive evidence of necessity.")

    add("MINIMAL_CONV_CARRIER", "Minimal Conv Carrier", f"The frozen development-only rule searched 19 nested/group candidates; none passed the four-probe strong gate. Full 24-layer Conv was therefore the **fallback**, not a successful small carrier. Selected set `{conv_selection['name']}`, layers `{conv_selection['conv_layers']}`, SHA-256 `{conv_selection['conv_layer_set_hash']}`. The full set failed development (L2 {f(conv['development'][conv['development'].condition=='FULL_CONV'].relative_l2_to_donor.median())}) but passed validation (L2 {f(conv['validation'][conv['validation'].condition=='FULL_CONV'].relative_l2_to_donor.median())}). Validation-only passing subsets were not promoted or searched adaptively. Neither localized-small-set V30-F nor distributed-necessary V30-G is established.")

    effect_rows = [(role,adj["REC_conditional_effect"][role]["states"],adj["REC_conditional_effect"][role]["REC_improves_Conv_relative_l2_count"],f(adj["REC_conditional_effect"][role]["Conv_only_median_relative_l2"]),f(adj["REC_conditional_effect"][role]["REC_Conv_median_relative_l2"]),f(adj["REC_conditional_effect"][role]["REC_only_median_relative_l2"]),f(adj["REC_conditional_effect"][role]["median_relative_l2_improvement"]),f(adj["REC_conditional_effect"][role]["median_interaction_ratio"])) for role in ROLES]
    add("REC_CONV_CONDITIONAL_EFFECT", "REC–Conv Conditional Effect", f"Exact native Y00 recipient, Y10 REC donor, Y01 Conv donor, Y11 REC+Conv donor were evaluated on the same frozen 10 states per role. Interaction is ‖Y11−Y10−Y01+Y00‖ / ‖donor−recipient‖, not an additivity assumption.\n\n{table(('role','states','REC improves count','Conv L2','REC+Conv L2','REC-only L2','paired median gain','interaction ratio'),effect_rows)}\nREC adds a reproducible conditional refinement (20/20 paired L2 reductions) while REC alone remains weak. This supports V30-H in this endpoint; it does not make REC an independent carrier or turn the development REC+Conv subset into a passed compact coordinate.")

    state_rank_rows=[(n,x['rows'],x['r90'],x['r95'],x['r99']) for n,x in rank120['state_count_scaling'].items()]
    token_rank_rows=[(n,x['rows'],x['r90'],x['r95'],x['r99']) for n,x in rank['train_token_pairs_per_state_scaling'].items()]
    add("WRITE_RANK_SCALING", "Write Rank Scaling", f"Centered pooled natural REC+Conv contrast ranks are descriptive, not causal-state dimensions. The 90-state×8-pair original fit had r90/r95/r99 = {fit['models']['GLOBAL']['ranks']['r90']}/{fit['models']['GLOBAL']['ranks']['r95']}/{fit['models']['GLOBAL']['ranks']['r99']}.\n\nBalanced state scaling (8 TRAIN pairs/state):\n\n{table(('states','rows','r90','r95','r99'),state_rank_rows)}\nOriginal-fit token coverage scaling (90 states):\n\n{table(('pairs/state','rows','r90','r95','r99'),token_rank_rows)}\nThe 120-state point uses 30 extra frozen development states and deterministic prewrite-eligible TRAIN pairs selected **after development response observation**; it is explicitly supplemental and was never used for any causal basis, transport fit, gate or finalist. State r95 growth slows after 80, but token coverage 1→8 still grows 24→131; the exact write geometry is not shown saturated. Behavioral k64 failure is a separate causal observation.")

    control_rows=[]
    for role in ROLES:
        for condition in ("EXACT_REC_CONV","GLOBAL_k64","RANDOM_SAME_NORM","SHUFFLED_COORDINATE","SIGN_FLIPPED","WRONG_STATE_COORDINATE","WRONG_TOKEN_COORDINATE","AMPLITUDE_0_5","AMPLITUDE_1_5"):
            z=controls[role][controls[role].condition==condition]
            control_rows.append((role,condition,f(z.donor_cosine.median()),f(z.magnitude_ratio.median()),f(z.relative_l2_to_donor.median())))
    add("NATURAL_VS_ARTIFICIAL_COORDINATES", "Natural versus Artificial Coordinates", f"On the predeclared 10-state/role Conv-profile panel, the TRAIN-only global k64 approximation was compared with exact natural REC+Conv, same-norm random tensor, shuffled coordinate, sign flip, wrong-source-state coordinate, wrong-token coordinate and α=.5/1.5 amplitudes. Random and shuffled reconstructions are deliberately off-manifold controls; their failure cannot adjudicate natural-write existence.\n\n{table(('role','condition','cosine','magnitude','relative L2'),control_rows)}\nStructured but inadequate global coordinates outperform random/shuffled/wrong-token controls. Wrong-state global coordinates being similar to target global coordinates does not establish causally adequate transport. Amplitude and sign do not exhibit a qualified linear control law; no per-coordinate semantic label is assigned.")

    add("STRICT_WRITE_INTERFACE_AUDIT", "Strict Write Interface Audit", f"Each comparison starts from the same architecture-native prefix cache; design records incoming REC/Conv/KV SHA-256 channel hashes and prefix-token hashes for all 255 frozen states. Current token forks finish naturally before any transplant. REC+Conv comprises all 24 recurrent/Conv layers (48 tensors; 13,369,344 float32 contrast components); KV contains 8 appended attention slots and is not silently relabeled as absent memory. Exact Conv depth writeback verifies donor equality of touched fields and recipient equality of untouched fields for each tested condition. Global and local `inject` clone the native recipient and alter only REC+Conv at identical cache length; these PCA realizations are off-manifold approximations, not exact native donor writes. The four-probe next-token sequence and current readout are fixed before transplant. Per-row native cache hashes and contrast hashes are in Parquet/JSON; large tensor matrices/bases remain outside Git under `{SCRATCH}` with committed hashes. No V1–V29 record was overwritten, and V30 independent-final responses were not opened. The 120-state rank supplement is descriptive only.")

    stage_rows=[]
    for fp in sorted((root/'artifacts').glob('transferable_natural_writes_v30*.freeze.json')):
        z=json.loads(fp.read_text())
        stage_rows.append((fp.name,z.get('freeze_digest','')[:16],sha256_file(fp)[:16]))
    add("EXECUTION_MANIFEST", "Execution Manifest", f"Parent Git commit `{json.loads((root/'artifacts/transferable_natural_writes_v30.freeze.json').read_text())['parent_commit']}`; V30 protocol/config base frozen before current-token writes. Panels: 25 calibration (prewrite only), 120 development, 60 validation, 50 independent final sealed. The execution plan freezes 90 fit states, 30 development holdout, 10 seen-state TOKEN-OOD tests, 60 validation targets, 8/16/32/64 dimensions, four probes, three causal axes, five LOFO models, transport rules, Conv groups and thresholds. A pre-response transport-source amendment reduced 85 sources to 5 without changing target pairs. An interrupted CPU basis fit produced no response record; the same frozen matrix/Gram was used for streamed GPU basis multiplication. A later 120-state spectrum supplement is explicitly descriptive.\n\n{table(('freeze file','digest prefix','file hash prefix'),stage_rows)}\nFormal final-opening decision `{final['reason']}`; final-opened=`{final['final_opened']}`; final decision file SHA-256 `{sha256_file(root/OUT/'final_opening_v30.json')}`. Frozen authorization: H2 true, H3 false, dynamic search false, autonomous controller false, cross-model replication false. Machine records are under `results/v30/processed/`; off-repository scratch is not in Git.")

    answers = [
        "1. V29 32/64D same-background success did not survive the expanded held-out-token OOD panel.",
        "2. Pooled exact write rank is not demonstrated saturated; token coverage r95 grows 24→131 for 1→8 pairs/state.",
        "3. A fixed global k64 does not pass unseen-state causal gates.",
        "4. It fails unseen token contrasts even on seen states.",
        "5. It fails joint unseen state+token OOD in development and validation.",
        "6. Five LOFO tests do not satisfy ≥4/5 causal replication.",
        "7. k32 is insufficient on all primary OOD axes.",
        "8. k64 is also insufficient on token/joint OOD.",
        "9. Required tested behavioral k exceeds 64 or requires a different representation; no minimal larger k was measured.",
        "10. Local 32D subspaces overlap strongly descriptively (median principal cosine ≈0.986 development).",
        "11. Some directions rotate (median maximum principal angle ≈41° development); geometry alone is not state dependence.",
        "12. The target local oracle is not materially better enough to pass held-out-token causal gates.",
        "13. Source→target coordinates were transported by TRAIN-only ridge/Procrustes maps.",
        "14. Their target-native future response did not pass the strong gate.",
        "15. No: held-out token contrasts fail for transport and target local oracle.",
        "16. No ≥4/5-family transported causal qualification.",
        "17. No fixed global basis is sufficient in this tested k≤64 regime.",
        "18. A state-dependent atlas is not established, because its local causal ceiling fails.",
        "19. V29 local same-background compact effects remain; V30 cannot confirm local-only compact sufficiency for new tokens.",
        "20. A full 24-single-layer Conv profile is recorded; no single layer passes the full causal claim.",
        "21. No development-selected small Conv layer subset passes; the frozen fallback is all 24 layers.",
        "22. Distributed necessity is not established, because even full Conv fails development four-probe gate.",
        "23. REC improves Conv conditional fidelity in all 20 profiled states across roles.",
        "24. REC×Conv interaction is material descriptively (median ratio ≈0.21 in both roles).",
        "25. KV-only remains weak for this tested future endpoint, not generally unimportant.",
        "26. Exact REC+Conv transfers across four probes; compact k32/64 does not.",
        "27. Exact REC+Conv is donor-directed at h2/h4, but h4 development misses the strong L2 gate; compact write remains unqualified.",
        "28. Structured global coordinates beat random/shuffled/wrong-token controls but fail the natural donor gate.",
        "29. α=.5/1.5 effects are measured and nonlinear; no qualified linear amplitude law.",
        "30. Sign flip does not show odd-symmetric causal control.",
        "31. V30-A false.", "32. V30-B false.", "33. V30-C false.", "34. Token-OOD generalization false.", "35. Family-OOD generalization false.", "36. Conv depth localization false; distributed necessity also not proven.", "37. Cross-model replication not authorized.",
    ]
    add("SCIENTIFIC_ANSWERS", "Scientific Answers", "The answers are restricted to the frozen token library, hybrid model and causal endpoint.\n\n" + "\n".join(answers) + "\n\nNo 32D/64D **model state** or exact natural-write dimensionality is claimed.")

    outcome_rows=[(name,str(value)) for name,value in adj['outcomes'].items()]
    complete=f"""**Generality and Geometry of Transferable Natural Writes — Is There a Shared Causal Write Language Across States, Tokens, and Tasks?** Parent `c247c82020b265e9589493262562851234825113`; frozen base `{json.loads((root/'artifacts/transferable_natural_writes_v30.freeze.json').read_text())['freeze_digest']}`; adjudication `{verify_stage(root,'adjudication')['freeze_digest']}`.

The prospective 25/120/60/50 panels exclude V28/V29 states. A calibration-only 88-pair common-token library is split 72/8/8 TRAIN/VALIDATION/FINAL. The primary REC+Conv write contrast has 13,369,344 float32 components. Ninety fitting states × eight TRAIN pairs yield pooled r90/r95/r99 = 74/131/320; this is descriptive write geometry, **not** a model-state dimension. The supplemental 120-state r95=135 is descriptive only and did not enter causal fitting.

The decisive four-probe future-response test does **not** identify a shared ≤64D write language. In validation JOINT-OOD, exact REC+Conv native donor write has median cosine {f(g['validation'][(g['validation'].axis=='JOINT_OOD')&(g['validation'].condition=='EXACT_REC_CONV')].donor_cosine.median())}, relative L2 {f(g['validation'][(g['validation'].axis=='JOINT_OOD')&(g['validation'].condition=='EXACT_REC_CONV')].relative_l2_to_donor.median())}; global k64 has {f(g['validation'][(g['validation'].axis=='JOINT_OOD')&(g['validation'].condition=='GLOBAL_k64')].donor_cosine.median())}/{f(g['validation'][(g['validation'].axis=='JOINT_OOD')&(g['validation'].condition=='GLOBAL_k64')].relative_l2_to_donor.median())}. Target local-oracle k64 L2 {f(local['validation'][local['validation'].condition=='LOCAL_ORACLE_k64'].relative_l2_to_donor.median())} and TRAIN-only transported ridge k64 L2 {f(local['validation'][local['validation'].condition=='TRANSPORTED_LOCAL_RIDGE_k64'].relative_l2_to_donor.median())} also miss the predeclared ≤0.30 gate. Because even the target local ceiling fails, this cannot isolate an inadequate transport map or prove a rotating atlas. V29's same-background 32/64D result remains valid in its own narrower regime.

Five-family LOFO and seen-state TOKEN-OOD tests likewise fail compact causal gates. The Conv all-24-layer fallback does not replicate the strict four-probe gate in both roles, so neither small-layer localization nor distributed-depth necessity is confirmed. REC conditionally improves Conv donor fidelity in 10/10 development and 10/10 validation states; REC-only remains weak. KV-only remains weak for this immediate future endpoint without denying its history-storage role. Natural exact writes outperform artificial controls. h2/h4 and amplitude/sign values are reported without promoting a failed compact model.

{table(('formal outcome','supported'),outcome_rows)}
No finalist qualified. The 50 independent-final states and TOKEN_FINAL contrasts were **not opened**; final-opening freeze `{verify_stage(root,'final_opening')['freeze_digest']}`. `H2_REMAINS=TRUE`; `H3_AUTHORIZED=FALSE`; `DYNAMIC_STATE_SEARCH_AUTHORIZED=FALSE`; `AUTONOMOUS_CONTROLLER_AUTHORIZED=FALSE`; `CROSS_MODEL_REPLICATION_AUTHORIZED=FALSE`. The original G/A/L trichotomy is incomplete under this expanded token library: a fourth possibility is that the tested ≤64D span omits token-dependent natural-write directions. Full machine records, hashes, reports and limitations follow in the separate V30 files and `V30_ALL_REPORTS.md`."""
    add("COMPLETE_REPORT", "Complete Report", complete)

    if set(reports) != set(ORDER):
        raise RuntimeError("V30 report set mismatch")
    for name in ORDER:
        (report_dir / f"V30_{name}.md").write_text(reports[name])
    bundle_parts=["# V30 — All Reports in One File\n\nAll 21 V30 report files are reproduced verbatim below in frozen report order. The individual files are authoritative.\n"]
    for j,name in enumerate(ORDER,1):
        path=report_dir/f"V30_{name}.md"
        bundle_parts.append(f"\n---\n\n<!-- {j:02d}: {path.name}; sha256={sha256_file(path)} -->\n\n{path.read_text()}")
    bundle_path=report_dir/'V30_ALL_REPORTS.md'
    bundle_path.write_text(''.join(bundle_parts))
    final_report=report_dir/'FINAL_REPORT.md'
    existing=final_report.read_text()
    if '# V30 Complete Report' in existing:
        raise RuntimeError('V30 already appended to cumulative FINAL_REPORT.md')
    final_report.write_text(existing.rstrip()+'\n\n---\n\n'+reports['COMPLETE_REPORT'])
    gram=np.load(SCRATCH/'train_write_gram_v30.npy')
    eig=np.maximum(np.linalg.eigvalsh(gram-gram.mean(0)[None,:]-gram.mean(1)[:,None]+gram.mean()),0)[::-1]
    npz=root/OUT/'v30_spectra.npz'
    np.savez_compressed(npz,global_singular_values=np.sqrt(eig).astype(np.float32),state_counts=np.array([10,20,40,80,120]),state_r95=np.array([rank120['state_count_scaling'][str(x)]['r95'] for x in (10,20,40,80,120)]),pairs_per_state=np.array([1,2,4,8]),token_r95=np.array([rank['train_token_pairs_per_state_scaling'][str(x)]['r95'] for x in (1,2,4,8)]))
    indexed=[]
    indexed += [root/'configs/transferable_natural_writes_v30.yaml',root/'src/jclosure/protocol_v30.py',root/'tests/test_v30_protocol.py']
    indexed += sorted((root/'src/jclosure/experiments').glob('*v30.py'))
    indexed += sorted((root/'artifacts').glob('transferable_natural_writes_v30*.freeze.json'))
    indexed += sorted((root/OUT).glob('*'))
    indexed += [report_dir/f'V30_{name}.md' for name in ORDER]
    indexed += [bundle_path,final_report]
    indexed=[x for x in indexed if x.is_file()]
    integrity={"schema_version":30,"protocol":"transferable_natural_writes_v30","entries":{str(x.relative_to(root)):sha256_file(x) for x in indexed},"entry_count":len(indexed),"independent_final_opened":False,"scratch_tensor_files_committed":False,"historical_final_opened":False}
    index_path=root/OUT/'v30_integrity_index.json'
    write_json_atomic(index_path,integrity)
    inputs=[SOURCE,str(index_path.relative_to(root)),str(bundle_path.relative_to(root)),str(final_report.relative_to(root)),str(npz.relative_to(root)),'artifacts/transferable_natural_writes_v30_adjudication.freeze.json']+[f'reports/V30_{name}.md' for name in ORDER]
    fr=stage_freeze(root,'reports',inputs,{"integrity_index_sha256":sha256_file(index_path),"all_reports_sha256":sha256_file(bundle_path),"complete_report_sha256":sha256_file(report_dir/'V30_COMPLETE_REPORT.md'),"report_count":len(ORDER),"independent_final_opened":False})
    return {"freeze_digest":fr['freeze_digest'],"report_count":len(ORDER),"bundle_sha256":sha256_file(bundle_path),"integrity_entries":len(indexed),"final_opened":False}


if __name__ == '__main__':
    print(json.dumps(run(Path.cwd()),indent=2))
