"""Generate V35 topic reports, complete report, all-in-one edition and append-only history."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v35 import verify, verify_stage

OUT = Path("results/v35/processed")
TOPICS = (
    "HIGH_LEVEL_RECONFIRMATION", "RECURRENT_READ_INTERFACE_AUDIT",
    "SINGLE_QUARTILE_MEDIATION", "EARLY_TO_LATE_CUMULATIVE", "LATE_TO_EARLY_CUMULATIVE",
    "LEAVE_QUARTILE_OUT", "QUARTILE_PAIR_MEDIATION", "COMPLEMENTARITY_REDUNDANCY",
    "SERIAL_MEDIATION", "DEPTH_CAUSAL_FLOW", "REFINEMENT_AUTHORIZATION",
    "REFINED_LAYER_WINDOWS", "CROSS_MODEL_DEPTH_PROFILE", "TASK_FAMILY_DEPTH_PROFILE",
    "TOKEN_CATEGORY_DEPTH_PROFILE", "CONV_DEPENDENCE_ACROSS_DEPTH", "REC_ONLY_DEPTH_CONTROL",
    "MULTI_HORIZON_DEPTH_MEDIATION", "STRICT_INTERFACE_AUDIT", "EXECUTION_MANIFEST",
    "SCIENTIFIC_ANSWERS", "COMPLETE_REPORT",
)
QUARTILES = ("Q1", "Q2", "Q3", "Q4")
ROLES = ("development", "validation", "independent_final")


def load(root, name):
    return json.loads((root / OUT / name).read_text())


def maybe(root, name):
    path = root / OUT / name
    return json.loads(path.read_text()) if path.exists() else None


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def yes(value):
    return "通过" if value else "未通过"


def table(headers, rows):
    return ("|" + "|".join(headers) + "|\n|" + "|".join("---" for _ in headers) + "|\n"
            + "\n".join("|" + "|".join(map(str, row)) + "|" for row in rows) + "\n")


def heading(title, body):
    return f"# V35 {title}\n\n{body.strip()}\n"


def run(root: Path):
    for stage in ("adjudication", "final_analysis", "depth_analysis_development",
                  "depth_analysis_validation", "final_opening", "control_plan"):
        verify_stage(root, stage)
    base = verify(root)
    cfg = base["config"]
    high = {role: load(root, f"high_level_{role}_v35.json") for role in ROLES}
    full = {role: load(root, f"full_gate_{role}_v35.json") for role in ROLES}
    depth = {role: load(root, f"depth_analysis_{role}_v35.json") for role in ROLES[:2]}
    plan = load(root, "development_plan_v35.json")
    opening = load(root, "final_opening_v35.json")
    final = load(root, "final_analysis_v35.json")
    adj = load(root, "v35_adjudication.json")
    interface = {key: load(root, f"interface_audit_{key}_v35.json") for key in ("Q", "F")}
    controls = {key: load(root, f"controls_{key}_validation_v35.json") for key in ("Q", "F")}
    flow = {key: load(root, f"flow_{key}_validation_v35.json") for key in ("Q", "F")}
    kv = {key: maybe(root, f"kv_control_{key}_validation_v35.json") for key in ("Q", "F")}
    horizons = {key: maybe(root, f"horizons_{key}_validation_v35.json") for key in ("Q", "F")}
    refinement = {key: {"selection": maybe(root, f"refinement_selection_{key}_v35.json"),
                        "development": maybe(root, f"refine_{key}_development_v35.json"),
                        "validation": maybe(root, f"refine_{key}_validation_v35.json")}
                  for key in ("Q", "F")}
    role_zh = {"development": "开发", "validation": "验证", "independent_final": "独立最终"}
    docs = {}
    high_rows = [(role_zh[role], key, x["rows"], fmt(x["positive_fraction"]),
                  fmt(x["median_reduction"]), fmt(x["median_alignment"]),
                  x["families_passing"], yes(x["pass"]))
                 for role in ROLES for key in ("Q", "F") for x in [high[role]["models"][key]]]
    high_table = table(("角色", "模型", "n", "正收益率", "修正减少", "残差对齐", "通过家族", "门槛"), high_rows)
    full_rows = [(role_zh[role], key, x["rows"], fmt(x["removed"]),
                  fmt(x["restored"]), fmt(x["cosine"]), x["families_passing"], yes(x["pass"]))
                 for role in ROLES for key in ("Q", "F") for x in [full[role]["models"][key]]]
    full_table = table(("角色", "模型", "n", "REM", "REST", "余弦", "通过家族", "门槛"), full_rows)
    docs["HIGH_LEVEL_RECONFIRMATION"] = heading("高层 REC+Conv 修正重确认",
        high_table + "\n冻结 V32–V34 端点定义、六个未来探针与校准标尺；开发集两模型先通过 ≥80% 正收益、修正减少 ≥0.20、对齐 ≥0.50、≥4/5 家族门槛，才启动深度实验。验证与独立最终未用于重新选择状态、token、阈值。")
    interface_rows = [(key, x["recurrent_layers"], yes(x["bitwise_baseline_replay"]),
                       yes(x["all_writeback_exact"]), x["group_hash"], x["condition_layers_hash"])
                      for key, x in interface.items()]
    docs["RECURRENT_READ_INTERFACE_AUDIT"] = heading("原生循环读出接口审计",
        table(("模型", "循环层", "原生重放", "校准写回", "相对深度组哈希", "条件层哈希"), interface_rows)
        + "\nQwen 使用原生 gated-delta mixer 输出，Falcon 使用原生 Mamba-2 mixer 输出；均为 V34 相同功能节点。每模型按原生有序循环层分成四个六层组，不将绝对层号视为跨架构同源。校准不接触正式角色响应。")

    def condition_table(names):
        rows = []
        for role in ROLES[:2]:
            for key in ("Q", "F"):
                for name in names:
                    x = depth[role]["models"][key]["conditions"][name]
                    rows.append((role_zh[role], key, name, fmt(x["removed_fraction"]),
                                 fmt(x["restored_fraction"]), fmt(x["reverse_correction_cosine"]),
                                 x["families_passing"], yes(x["pass"])))
        return table(("角色", "模型", "条件", "REM", "REST", "余弦", "通过家族", "强门槛"), rows)

    docs["SINGLE_QUARTILE_MEDIATION"] = heading("单四分位双向中介",
        condition_table(QUARTILES) + "\nREM 是从联合分支移除该读出组，REST 是向 Conv-only 分支插入该组；两者均以 REC 对 Conv-only 的附加修正收益为分母。未过强门槛的连续效应仍是部分贡献，不称该组为独立校正模块。")
    docs["EARLY_TO_LATE_CUMULATIVE"] = heading("早到晚累计曲线",
        condition_table(cfg["early_cumulative"]) + "\n按 E1=Q1、E2=Q1+Q2、E3=Q1+Q2+Q3、E4=全深度固定顺序累计。门槛列仅是每个子集的强双向门槛；渐增机制另需每一步的预定最小增益与五类任务复现。"
        + "\n" + table(("角色", "模型", "渐增组织"),
                       [(role_zh[role], key, yes(depth[role]["models"][key]["progressive_accumulation"]["pass"]))
                        for role in ROLES[:2] for key in ("Q", "F")]))
    docs["LATE_TO_EARLY_CUMULATIVE"] = heading("晚到早累计曲线",
        condition_table(cfg["late_cumulative"]) + "\n按 L1=Q4、L2=Q3+Q4、L3=Q2+Q3+Q4、L4=全深度固定顺序累计。正式晚半区判据同时要求 Q3+Q4 双向 ≥0.75、Q1+Q2 双向 ≤0.40、至少 0.20 的差值、≥4/5 家族，并在开发和验证重现。"
        + "\n" + table(("角色", "模型", "晚半区集中"),
                       [(role_zh[role], key, yes(depth[role]["models"][key]["late_consolidation"]["pass"]))
                        for role in ROLES[:2] for key in ("Q", "F")]))
    leave_rows = [(role_zh[role], key, name,
                   fmt(depth[role]["models"][key]["leaveout"][name]["median_loss_fraction"]))
                  for role in ROLES[:2] for key in ("Q", "F") for name in cfg["leave_out_conditions"]]
    docs["LEAVE_QUARTILE_OUT"] = heading("全读出移植下留一组必要性",
        table(("角色", "模型", "条件", "修正保真损失/高层收益"), leave_rows)
        + "\n从完整联合读出移植开始，仅把指定四分位恢复为 Conv-only 读出。损失为相对全移植的供体误差上升，报告连续值；没有预先冻结的单组必要性二值阈值，因此不把任一留一组结果单独宣称为必要模块。")
    pair_rows = [(role_zh[role], key, pair,
                  fmt(depth[role]["models"][key]["conditions"][pair]["removed_fraction"]),
                  fmt(depth[role]["models"][key]["conditions"][pair]["restored_fraction"]),
                  fmt(depth[role]["models"][key]["pair_interactions"][pair]["median_projection_to_full_correction"]),
                  yes(depth[role]["models"][key]["conditions"][pair]["pass"]))
                 for role in ROLES[:2] for key in ("Q", "F") for pair in cfg["pair_conditions"]]
    docs["QUARTILE_PAIR_MEDIATION"] = heading("六个预定深度配对",
        table(("角色", "模型", "配对", "REM", "REST", "交互投影", "强双向"), pair_rows)
        + "\n交互按输出向量 `Y_ij − Y_i − Y_j + Y_0` 逐状态计算，再投影到完整 REC 修正方向并取中位；这是精确四分支交互，不将 REM/REST 比例相加。全 15 个非空深度子集均预先声明，未组合搜索全部 24 层。")
    classification_rows = [(role_zh[role], key, pair,
                            yes(depth[role]["models"][key]["pairs"][pair]["complementary"]["pass"]),
                            yes(depth[role]["models"][key]["pairs"][pair]["redundancy"]["pass"]))
                           for role in ROLES[:2] for key in ("Q", "F") for pair in cfg["pair_conditions"]]
    docs["COMPLEMENTARITY_REDUNDANCY"] = heading("互补与冗余判定",
        table(("角色", "模型", "配对", "互补", "冗余"), classification_rows)
        + "\n互补要求配对 REM 与 REST 均比最强单组高 ≥0.10，且任一单组未独自恢复大部分修正。冗余要求两个单组双向均较大，配对增加 ≤0.10；任一方向缺失即不称冗余。两种分类均要求 ≥4/5 家族。")
    serial_rows = [(role_zh[role], key, pair,
                    fmt(x["values"]["conditional_late_gain"]),
                    fmt(x["values"]["removal_interaction"]),
                    fmt(x["values"]["interaction_projection"]), yes(x["pass"]))
                   for role in ROLES[:2] for key in ("Q", "F")
                   for pair, x in depth[role]["models"][key]["serial_pairs"].items()]
    docs["SERIAL_MEDIATION"] = heading("串行四分支中介",
        table(("角色", "模型", "早→晚", "条件性晚增益", "移除侧交互", "输出交互投影", "正式串行"), serial_rows)
        + "\n在同一 Conv-only 基底上精确构造早/晚 `C_C、C_J、J_C、J_J` 四分支。正式串行需要条件性晚增益、移除侧互证、正向交互及 ≥4/5 家族，不从先后顺序或激活范数推断串行。")
    flow_rows = [(key, q, kind, fmt(value)) for key in ("Q", "F")
                 for label, value in flow[key]["median_joint_vs_conv_l2_within_model"].items()
                 for q, kind in [label.split(":")]]
    docs["DEPTH_CAUSAL_FLOW"] = heading("深度边界因果流解释性轨迹",
        table(("模型", "边界", "张量", "joint−Conv L2 中位"), flow_rows)
        + "\n验证集每家族一个冻结状态、首个冻结探针，记录 recipient、Conv-only、joint 在四个边界的残差与原生读出哈希及差异。轨迹仅解释已通过的替换实验；原始 L2 不跨模型比较，也不作为正式中介证据。")
    auth_rows = [(key, ", ".join(plan["model_refinement"][key]["qualified_coarse_regions"]) or "无",
                  plan["model_refinement"][key]["selected_region"] or "未授权",
                  ", ".join(plan["model_refinement"][key]["three_layer_windows"]) or "—")
                 for key in ("Q", "F")]
    docs["REFINEMENT_AUTHORIZATION"] = heading("开发集三层窗口细化授权",
        table(("模型", "开发强粗区域", "预定细化区域", "三层窗口"), auth_rows)
        + "\n只有开发集的强四分位或相邻强配对可授权细化；每模型按预定优先级选一个区域、等分成三层窗口。窗口名中的 W 数字是该区域中的起始偏移，不是绝对层号。验证不得选择不同窗口；未做全 24 层单层搜索。")
    refined_rows = []
    for key in ("Q", "F"):
        selection = refinement[key]["selection"]
        if selection is None:
            refined_rows.append((key, "未授权", "—", "—", "—"))
            continue
        for name, score in selection["scores"].items():
            val = refinement[key]["validation"]
            validated = "未打开" if name != selection["selected_window"] or val is None else "仅此窗口执行"
            refined_rows.append((key, name, fmt(score["removed_fraction"]),
                                 fmt(score["restored_fraction"]), f"{yes(score['pass'])} / {validated}"))
    docs["REFINED_LAYER_WINDOWS"] = heading("固定三层窗口中介",
        table(("模型", "窗口", "开发 REM", "开发 REST", "开发强门槛 / 验证"), refined_rows)
        + "\n开发集测试预定窗口并冻结唯一候选；若无窗口达到双向强门槛，验证不打开窗口。即使粗区域强，也不能据此称其内部任一三层窗口充分。")
    profile_rows = [(side, fmt(x["spearman"]), fmt(x["pearson"]), fmt(x["cosine"]),
                     fmt(x["early_curve_l2"]), fmt(x["late_curve_l2"]))
                    for side, x in depth["validation"]["cross_model_profile"].items() if side in ("REM", "REST")]
    docs["CROSS_MODEL_DEPTH_PROFILE"] = heading("跨模型相对深度画像",
        table(("方向", "Spearman", "Pearson", "画像余弦", "早曲线 L2", "晚曲线 L2"), profile_rows)
        + f"\n四点画像的描述性相似门槛：{yes(depth['validation']['cross_model_profile']['descriptive_profile_similarity_pass'])}。正式共享层级仍以两模型相同定性因果门槛在开发与验证分别通过为准；相似系数不能替代双向干预。相似门槛未过也不自动证明架构特异，尤其四点相关值接近阈值时。比较仅用无量纲效应和相对深度。")
    family_rows = [(role_zh[role], key, family, q, fmt(values["removed_fraction"]),
                    fmt(values["restored_fraction"]))
                   for role in ROLES[:2] for key in ("Q", "F")
                   for family, conditions in depth[role]["models"][key]["family_depth_profile"].items()
                   for q, values in conditions.items() if q in QUARTILES or q == "Q3_Q4"]
    docs["TASK_FAMILY_DEPTH_PROFILE"] = heading("五类任务的深度画像",
        table(("角色", "模型", "任务族", "深度条件", "REM", "REST"), family_rows)
        + "\n保留布尔逻辑、模运算、短图遍历、状态转移和变量绑定各自的分布；正式 ≥4/5 家族门槛使用每家族独立的双向中位数。")
    category_rows = [(role_zh[role], key, category, q,
                      fmt(values["removed_fraction"]), fmt(values["restored_fraction"]))
                     for role in ROLES[:2] for key in ("Q", "F")
                     for category, conditions in depth[role]["models"][key]["token_category_depth_profile"].items()
                     for q, values in conditions.items()]
    docs["TOKEN_CATEGORY_DEPTH_PROFILE"] = heading("token 类别深度画像",
        table(("角色", "模型", "供体 token 类别", "四分位", "REM", "REST"), category_rows)
        + "\n类别由冻结设计中的供体 token 标注：词汇表层、功能词、数字、标点/结构及可用时的普通未解析类别。Falcon 若某类没有预先选中的 token，则保持缺失，不补抽、不外推；此项仅为二级描述性分析。")
    control_rows = [(key, q, fmt(v["joint_restore_median"]),
                     fmt(v["rec_only_restore_median"]),
                     fmt(v["joint_minus_rec_restore_median"]),
                     fmt(v["all_six_read_hashes_different_fraction"]))
                    for key in ("Q", "F") for q, v in controls[key]["quartiles"].items()]
    control_table = table(("模型", "四分位", "joint REST", "REC-only REST", "条件差", "6/6 读出哈希差比例"), control_rows)
    docs["CONV_DEPENDENCE_ACROSS_DEPTH"] = heading("Conv 条件依赖随深度的变化",
        control_table + "\n验证集冻结十状态、原生 recipient KV；将 donor REC+donor Conv 与 donor REC+recipient Conv 产生的原生读出分别插入同一个 Conv-only 目标。差值为方向性辅助证据；小样本不单独定位唯一的 Conv×REC 门控。")
    docs["REC_ONLY_DEPTH_CONTROL"] = heading("REC-only 深度对照",
        control_table + "\nREC-only 源只替换 REC 状态、保留 recipient Conv；joint 源同时带 donor REC 与 donor Conv。各四分位均精确写回六个同 token 探针。若 joint 强于 REC-only，可称该区域对 Conv 背景条件敏感，不能从哈希差本身推出足够的未来修正。REC-only 交叉状态可能偏离自然轨迹；负恢复比例不可独立证明唯一 Conv 门控。")
    horizon_rows = [(key, label, fmt(v["removed"]), fmt(v["restored"]), fmt(v["cosine"]))
                    for key in ("Q", "F") if horizons[key] for label, v in horizons[key]["median"].items()]
    horizon_note = ("仅在开发和验证共享层级合格后启动。每模型每家族一个冻结状态，h1 精确补丁；h2/h4 后续 token 不再补丁，自然演化。"
                    if horizon_rows else "未启动：没有开发与验证共同通过的较小/结构化深度机制；不能以 V34 探索性多步结果替代 V35 数据。")
    docs["MULTI_HORIZON_DEPTH_MEDIATION"] = heading("多步深度中介",
        (table(("模型", "条件/步数", "REM", "REST", "余弦"), horizon_rows) if horizon_rows else "无 h2/h4 正式运行记录。\n")
        + "\n" + horizon_note + "动态重建 h2 的比较属于可选二级实验，V35 未在未冻结修订的情况下启动。")
    strict_rows = [(key, role_zh[role], load(root, f"depth_{key}_{role}_v35.json")["audit_rows"],
                    yes(load(root, f"depth_{key}_{role}_v35.json")["all_exact_writeback"]))
                   for role in ROLES[:2] for key in ("Q", "F")]
    docs["STRICT_INTERFACE_AUDIT"] = heading("严格写回与污染审计",
        table(("模型", "角色", "逐探针条件审计行", "请求=实现"), strict_rows)
        + "\n每一次写入保存 source-map、requested、realized、native 与逐层哈希；请求与实现不一致会使运行失败。相同状态/位置/六探针、原生 recipient KV 与历史排除均由机器记录核对。独立最终仅在开发—验证判定与 finalist 哈希封存之后打开。")
    model_rows = [(key, spec["revision"], ", ".join(spec["weight_sha256"].values()),
                   spec["tokenizer_sha256"], spec["config_sha256"])
                  for key, spec in base["model_specs"].items()]
    kv_rows = [(key, condition, fmt(values["removed"]), fmt(values["restored"]))
               for key in ("Q", "F") if kv[key] for condition, values in kv[key]["median"].items()]
    docs["EXECUTION_MANIFEST"] = heading("执行清单与哈希",
        table(("模型", "revision", "权重 SHA-256", "tokenizer SHA-256", "config SHA-256"), model_rows)
        + f"\n父提交 `{base['parent_commit']}`；基础协议 `{base['freeze_digest']}`；开发计划 `{verify_stage(root,'development_plan')['freeze_digest']}`；最终开放 `{verify_stage(root,'final_opening')['freeze_digest']}`。每模型 20/80/40/40 全新语义配对状态，排除 V28–V34 正式状态。机器 JSON/Parquet/NPZ、设计 token/探针/深度哈希、逐次写回哈希与完整索引在 `results/v35/processed/`。\n\n"
        + ("供体 KV 小子集敏感性：\n\n" + table(("模型", "深度条件", "REM", "REST"), kv_rows) if kv_rows else "供体 KV 小子集未运行。"))

    shared = depth["validation"]["shared_qualitative_both_roles"]
    qv, fv = depth["validation"]["models"]["Q"], depth["validation"]["models"]["F"]
    strongest = {key: {metric: max(QUARTILES, key=lambda q: depth["validation"]["models"][key]["conditions"][q][metric])
                       for metric in ("removed_fraction", "restored_fraction")} for key in ("Q", "F")}
    def cv(key, condition, col):
        return depth["validation"]["models"][key]["conditions"][condition][col]
    def both(condition, col):
        return f"Q={fmt(cv('Q', condition, col))}、F={fmt(cv('F', condition, col))}"
    outcome = adj["formal_outcomes"]
    answers = [
        f"1. 新面板全深度读出复现：{yes(adj['full_depth_reconfirmed_all_roles'])}；三角色两模型详情见表。",
        f"2. Q1 验证 REM/REST：{both('Q1','removed_fraction')} / {both('Q1','restored_fraction')}。",
        f"3. Q2 验证 REM/REST：{both('Q2','removed_fraction')} / {both('Q2','restored_fraction')}。",
        f"4. Q3 验证 REM/REST：{both('Q3','removed_fraction')} / {both('Q3','restored_fraction')}。",
        f"5. Q4 验证 REM/REST：{both('Q4','removed_fraction')} / {both('Q4','restored_fraction')}。",
        f"6. 验证 REM 最大的四分位：Q={strongest['Q']['removed_fraction']}、F={strongest['F']['removed_fraction']}。",
        f"7. 验证 REST 最大的四分位：Q={strongest['Q']['restored_fraction']}、F={strongest['F']['restored_fraction']}。",
        f"8. 晚半区 Q3+Q4 的强双向门槛：{yes('Q3_Q4' in shared['strong_pairs'])}；正式晚集中另须更严格的 0.75 条件。",
        f"9. 早半区 Q1+Q2 的强双向门槛：{yes('Q1_Q2' in shared['strong_pairs'])}。",
        f"10. 两模型开发/验证共同的渐增机制：{yes(shared['progressive'])}；两方向每步增益及家族规则均计入。",
        f"11. 晚到早曲线是否正式更快：{yes(shared['late'])}；不能只凭一条曲线的形状。",
        f"12. Q3+Q4 验证 REM/REST：{both('Q3_Q4','removed_fraction')} / {both('Q3_Q4','restored_fraction')}。",
        "13. 留一四分位的全移植损失逐模型逐家族报告；未冻结单组必要性二值阈值，因此不宣称某一组是唯一必要模块。",
        f"14. Q3 与 Q4 的互补性：{yes('Q3_Q4' in shared['complementary_pairs'])}；依据双向配对超额及精确四分支交互。",
        f"15. Q3 与 Q4 的冗余性：{yes('Q3_Q4' in shared['redundant_pairs'])}；未用单侧恢复值替代移除互证。",
        f"16. 早读出是否使晚修正产生条件性增益：共同合格配对为 {', '.join(shared['serial_pairs']) or '无'}。",
        f"17. 正式串行中介：{yes(bool(shared['serial_pairs']))}；不能从深度顺序本身推断。",
        f"18. 固定三层窗口跨模型合格：{yes(all(refinement[k]['selection'] and refinement[k]['selection']['selected_window'] for k in ('Q','F')))}；只有开发选定窗口才可能进入验证。",
        f"19. 相对晚深度集中：{yes(shared['late'])}，受双向 0.75/0.40/家族门槛约束。",
        f"20. 逐层渐增组织：{yes(shared['progressive'])}，不要求线性但要求连续增益。",
        f"21. 两模型是否出现相同合格组织：{yes(bool(adj['shared_development_validation_patterns']))}；详见共同模式清单。",
        f"22. 四点相对深度画像的描述性相似：{yes(depth['validation']['cross_model_profile']['descriptive_profile_similarity_pass'])}；不替代因果门槛。",
        f"23. 架构特异的正式层级：{yes(outcome['V35-G_ARCHITECTURE_SPECIFIC_HIERARCHY'])}；不把数值差异自动称不同机制。",
        "24. Conv 条件依赖见各四分位 joint 与 REC-only 的同基底 REST 差；冻结十状态辅助对照，不能单独定位唯一门控。",
        "25. 合格粗区域的 REC-only 是否弱于 joint：逐四分位数值见 REC-only 对照；只作小样本条件性证据。",
        "26. token 类别依赖为描述性画像；缺失类别不补抽，不作正式类别差异检验。",
        "27. 五类任务的 REM/REST 分开列出，正式判定要求 ≥4/5 家族而非只看整体中位。",
        f"28. 合格深度机制对 h2/h4 的影响：{'已执行冻结五状态/模型小子集，见多步报告' if horizon_rows else '未授权，未执行'}。",
        "29. 是否每 token 重建：h2 重复补丁可选实验未获冻结修订，V35 不作结论。",
        f"30. V35-A 晚集中确认：{yes(outcome['V35-A_LATE_READ_CONSOLIDATION_CONFIRMED'])}。",
        f"31. V35-B 渐增确认：{yes(outcome['V35-B_PROGRESSIVE_READ_ACCUMULATION_CONFIRMED'])}。",
        f"32. V35-C 互补 / V35-D 冗余：{yes(outcome['V35-C_COMPLEMENTARY_MULTI_STAGE_MEDIATION'])} / {yes(outcome['V35-D_REDUNDANT_READ_MEDIATION'])}。",
        f"33. V35-E 串行：{yes(outcome['V35-E_SERIAL_HIERARCHICAL_CORRECTION'])}。",
        f"34. V35-F 共享 / V35-G 架构特异：{yes(outcome['V35-F_SHARED_HIERARCHICAL_ORGANIZATION'])} / {yes(outcome['V35-G_ARCHITECTURE_SPECIFIC_HIERARCHY'])}。",
        f"35. V35-H 局部区域最终确认：{yes(outcome['V35-H_LOCALIZED_READ_REGION'])}；开发/验证的较大粗配对支持另行列明。",
        f"36. V35-I 仅全深度可恢复：{yes(outcome['V35-I_FULL_DEPTH_READ_CUT_ONLY'])}；`FULL_DEPTH_ONLY` 是共同最终候选的回退名称，不证明每个模型内部均无较小有效组织。",
        f"37. 第三模型冻结复制对象：`{adj['third_model_replication_freeze']}`；未执行第三模型或应用基准。",
    ]
    if len(answers) != 37:
        raise RuntimeError("V35 scientific answers not one-to-one")
    outcome_rows = [(name, yes(value)) for name, value in adj["formal_outcomes"].items()]
    docs["SCIENTIFIC_ANSWERS"] = heading("37 项科学问题逐项回答",
        "\n\n".join(answers) + "\n\n" + table(("正式结局", "判定"), outcome_rows))
    main = (f"独立最终预先选定 `{opening['selected_mechanism']}`；两模型确认={yes(final['both_pass'])}。"
            f" V35 正式结局：{', '.join(name for name, passed in adj['formal_outcomes'].items() if passed) or '无确认项'}。")
    docs["COMPLETE_REPORT"] = heading("Complete Report — Hierarchical Decomposition of Recurrent-Read Mediation",
        "**How Is REC–Conv Correction Distributed Across Recurrent Depth?**\n\n"
        + main + "\n\n## 高层与全读出门槛\n\n" + high_table + "\n" + full_table
        + "\n## 四分位与共同机制\n\n" + condition_table(QUARTILES + ("Q1_Q2", "Q3_Q4"))
        + f"\n共同开发/验证：晚集中={yes(shared['late'])}；渐增={yes(shared['progressive'])}；"
          f"互补配对={', '.join(shared['complementary_pairs']) or '无'}；"
          f"串行配对={', '.join(shared['serial_pairs']) or '无'}；"
          f"强子集={', '.join(shared['strong_quartiles'] + shared['strong_pairs']) or '无'}。\n\n"
        + "## 独立最终与边界\n\n"
        + f"最终条件 `{', '.join(opening['selected_conditions'])}`；家族规则 ≥4/5；最终结果={yes(final['both_pass'])}。"
          "`FULL_DEPTH_ONLY` 作为共同候选回退，不等于 V35-I 已成立；两模型若各有不同的开发—验证合格模式，仍不能说其内部全深度不可分。"
          "其他已测机制不得由完整读出=1 反推每层必要。开发细化、REC-only/Conv、供体 KV、h2/h4 与 token/任务异质性见各专题报告；小子集证据不改变主门槛。"
        + f" 应用实验授权={adj['application_experiments_authorized']}，但未测应用收益。第三模型冻结对象 `{adj['third_model_replication_freeze']}`。"
          "V1–V34 历史记录未改、旧独立最终集未重开；V35 机器证据与 SHA-256 索引在 `results/v35/processed/`。\n\n"
        + table(("正式结局", "判定"), outcome_rows))
    if set(docs) != set(TOPICS):
        raise RuntimeError(f"V35 report topics mismatch: {set(TOPICS) - set(docs)}")
    report_dir = root / "reports"
    for name in TOPICS:
        path = report_dir / f"V35_{name}.md"
        if path.exists():
            raise RuntimeError(f"V35 report already exists: {path}")
    combined_path = report_dir / "V35_ALL_REPORTS.md"
    if combined_path.exists():
        raise RuntimeError("V35 all-reports edition already exists")
    cumulative = report_dir / "FINAL_REPORT.md"
    old = cumulative.read_text(encoding="utf-8")
    if "## V35 — Hierarchical Decomposition" in old:
        raise RuntimeError("V35 already appended to FINAL_REPORT")
    for name in TOPICS:
        (report_dir / f"V35_{name}.md").write_text(docs[name].rstrip() + "\n", encoding="utf-8")
    combined = "# V35 All Reports\n\n本文件逐一完整收录 V35 全部 22 份专题/综合报告，与各同名文件内容一致。\n"
    for name in TOPICS:
        combined += f"\n---\n\n<!-- V35_{name}.md -->\n\n" + docs[name].rstrip() + "\n"
    combined_path.write_text(combined, encoding="utf-8")
    cumulative.write_text(old.rstrip() + "\n\n## V35 — Hierarchical Decomposition\n\n"
        + main + " 第三模型冻结对象：`" + adj["third_model_replication_freeze"]
        + "`。详见 `reports/V35_COMPLETE_REPORT.md` 与 `reports/V35_ALL_REPORTS.md`。\n", encoding="utf-8")
    return {"reports": len(TOPICS) + 1, "all_reports": "reports/V35_ALL_REPORTS.md",
            "cumulative_appended": True}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2, ensure_ascii=False))
