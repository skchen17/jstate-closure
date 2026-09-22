"""Generate V34 per-topic, complete, combined and cumulative reports from sealed records."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v34 import verify, verify_stage

OUT = Path("results/v34/processed")
STAGES = ("TRUE_UPDATE", "TRANSFORMED_CONTROL", "POSTCONV_INPUT", "RECURRENT_READ", "RESIDUAL_INTEGRATION")
ROLES = ("development", "validation", "independent_final")
NAMES = (
    "REPLICATION_RECONFIRMATION", "FUNCTIONAL_STAGE_MAPPING", "EXACT_INSTRUMENTATION_AUDIT",
    "UPDATE_TERM_MEDIATION", "GATE_DECAY_MEDIATION", "POSTCONV_INPUT_MEDIATION",
    "RECURRENT_READ_MEDIATION", "RESIDUAL_INTEGRATION_MEDIATION", "MULTI_STAGE_PIPELINE_MEDIATION",
    "MEDIATION_OVERLAP", "CONTEXT_SPECIFIC_MEDIATOR", "CONV_DEPENDENCE",
    "REC_ONLY_MICRO_CONTROL", "FUNCTIONAL_MEDIATION_PROFILE", "CROSS_MODEL_MEDIATOR_COMPARISON",
    "DEPTH_LOCALIZATION", "MULTI_HORIZON_MEDIATION", "STRICT_INTERFACE_AUDIT",
    "EXECUTION_MANIFEST", "SCIENTIFIC_ANSWERS", "COMPLETE_REPORT",
)


def load(root, name):
    return json.loads((root / OUT / name).read_text())


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def table(headers, rows):
    return "|" + "|".join(headers) + "|\n|" + "|".join("---" for _ in headers) + "|\n" + "\n".join("|" + "|".join(map(str, row)) + "|" for row in rows) + "\n"


def write(path, content):
    if path.exists():
        raise RuntimeError(f"V34 report already exists, refusing overwrite: {path}")
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def run(root: Path):
    for seal in ("stage_analysis_development", "stage_analysis_validation", "final_analysis", "adjudication"):
        verify_stage(root, seal)
    cfg = verify(root)["config"]
    high = {r: load(root, f"high_level_{r}_v34.json") for r in ROLES}
    stages = {r: load(root, f"stage_analysis_{r}_v34.json") for r in ROLES[:2]}
    final = load(root, "final_analysis_v34.json")
    adj = load(root, "v34_adjudication.json")
    controls = {m: load(root, f"followup_controls_{m}_validation_v34.json") for m in ("Q", "F")}
    depth = {m: load(root, f"depth_{m}_validation_v34.json") for m in ("Q", "F")}
    horizons = {m: load(root, f"horizons_{m}_validation_v34.json") for m in ("Q", "F")}
    components = {m: load(root, f"components_{m}_validation_v34.json") for m in ("Q", "F")}
    audit = {m: load(root, f"interface_audit_{m}_v34.json") for m in ("Q", "F")}
    role_label = {"development": "开发", "validation": "验证", "independent_final": "独立最终"}
    hrows = []
    for role in ROLES:
        for model in ("Q", "F"):
            x = high[role]["models"][model]
            hrows.append((role_label[role], model, x["rows"], fmt(x["positive_fraction"]),
                          fmt(x["median_reduction"]), fmt(x["median_alignment"]),
                          x["families_passing"], "通过" if x["pass"] else "未通过"))
    high_table = table(("角色", "模型", "n", "正收益率", "中位修正", "中位对齐", "家族", "门槛"), hrows)
    stage_rows = []
    for stage in STAGES:
        for role in ROLES:
            for model in ("Q", "F"):
                x = (stages[role]["models"][model][stage] if role != "independent_final"
                     else final["models"][model] if stage == final["selected_stage"] else None)
                if x is None:
                    continue
                stage_rows.append((stage, role_label[role], model, fmt(x.get("removed_fraction")),
                                   fmt(x.get("restored_fraction")), fmt(x.get("reverse_correction_cosine")),
                                   x.get("families_passing", "—"), "通过" if x["pass"] else x.get("status", "未通过")))
    stage_table = table(("功能阶段", "角色", "模型", "REM", "REST", "余弦", "家族", "判定"), stage_rows)
    mapping = table(("功能阶段", "Qwen 原生对象", "Falcon 原生对象", "正式身份"), (
        ("TRUE_UPDATE", "true delta / rank-one write", "true dBx", "写侧，可独立写入"),
        ("TRANSFORMED_CONTROL", "transformed g / beta", "transformed dt / decay", "写侧，可独立写入"),
        ("POSTCONV_INPUT", "post-Conv q/k/v", "post-Conv x/B/C", "写侧，可独立写入"),
        ("RECURRENT_READ", "recurrent mixer output", "SSM mixer output", "读侧，可独立写入"),
        ("RESIDUAL_INTEGRATION", "same mixer contribution at residual add", "same mixer contribution at residual add", "与读出节点别名，无法独立识别"),
    ))
    main = "**V34-A 与 V34-E 成立**：两模型同一读出功能阶段在开发、验证和独立最终集上通过双向因果中介门槛。"
    caution = ("这是对全 24 个循环层读出贡献的宽因果截面，不证明两模型采用相同的更新代数，"
               "也不能把与其别名的残差整合点另算一个独立中介。")
    evidence = ("判据相对于 REC 对 Conv-only 的附加修正收益，而非完整供体效应；"
                "所有正式写入保持同一 token/位置、原生 recipient KV、六个冻结探针，逐次记录 requested/realized 哈希。")
    docs = {}
    docs["REPLICATION_RECONFIRMATION"] = "# V34 高层效应重确认\n\n" + high_table + "\n开发集先在两模型通过 ≥80% 正收益、中位修正 ≥0.20、中位对齐 ≥0.50、≥4/5 家族门槛，才开启正式内部中介。验证和独立最终集的高层效应随后分别重现。定义沿用 V32/V33，未重调。\n"
    docs["FUNCTIONAL_STAGE_MAPPING"] = "# V34 功能阶段映射\n\n" + mapping + "\n比较的是功能角色，不是张量名称、形状或数学公式。Qwen 与 Falcon 使用不同的循环微代数；24 个循环层均进入全深度正式干预。第五阶段在两个实装中是第四阶段输出直接进入残差加法，没有中间可独立覆盖的对象。\n"
    docs["EXACT_INSTRUMENTATION_AUDIT"] = "# V34 精确仪器审计\n\n" + table(("模型", "校准层数", "原生重放 bitwise", "校准干预次数"), [(m, audit[m]["recurrent_layers_audited"], audit[m]["bitwise_baseline_replay"], len(audit[m]["interventions"])) for m in ("Q", "F")]) + "\n校准使用未进入正式集的状态；每一正式阶段两方向都在每个探针与层记录原生、请求、实现哈希。正式 Parquet 位于 `results/v34/processed/mediation_audit_*_v34.parquet`；完整精度与拓扑审计以原始记录为准。\n"
    stage_docs = {
        "UPDATE_TERM_MEDIATION": ("TRUE_UPDATE", "真实更新项只覆盖当前 token 的写入贡献，不等于完整新 REC 状态。Qwen 的反向插入在独立开发、验证中反而恶化供体误差；不能由重构 outgoing state 推出它是中介。"),
        "GATE_DECAY_MEDIATION": ("TRANSFORMED_CONTROL", "使用核实际消费的变换后 Qwen g/beta 与 Falcon dt/decay，不是原始投影 logits；两模型双向效果均远低于强门槛。"),
        "POSTCONV_INPUT_MEDIATION": ("POSTCONV_INPUT", "使用短 Conv 之后进入循环算子的确切输入，而不是 Conv 前投影。它们可双向精确写入，但恢复比例与方向未达到强门槛。"),
        "RECURRENT_READ_MEDIATION": ("RECURRENT_READ", "全 24 层状态条件化 mixer 输出双向替换通过各角色、各模型、五类任务；这是读侧因果截面。全层输出复制可以形成宽截面，不能单凭 1.00 的效果定位内部写入算法。"),
        "RESIDUAL_INTEGRATION_MEDIATION": ("RESIDUAL_INTEGRATION", "校准显示进入残差的最早可写循环贡献与 mixer 输出是同一个张量/节点；不存在更下游且仍只覆盖循环贡献的独立接口。故标记 ALIASED_NOT_SEPARATELY_IDENTIFIABLE，V34-F 不成立，不能重复计数 V34-E。"),
    }
    for name, (stage, note) in stage_docs.items():
        rows = [x for x in stage_rows if x[0] == stage]
        docs[name] = f"# V34 {stage} 中介\n\n" + table(("阶段", "角色", "模型", "REM", "REST", "余弦", "家族", "判定"), rows) + "\n" + note + "\n"
    component_rows = [(m, component, fmt(values["median_removed"]),
                       fmt(values["median_restored"]), fmt(values["median_cosine"]))
                      for m, record in components.items() for component, values in record["components"].items()]
    component_table = table(("模型", "原生子量", "REM", "REST", "余弦"), component_rows)
    docs["GATE_DECAY_MEDIATION"] += "\n冻结验证小子集的可分离子量：\n\n" + component_table + "\nFalcon dt 与由其确定的 decay 是耦合变量，未把派生 decay 伪装成独立控制点。子量实验是探索性，不重定义正式联合阶段。\n"
    docs["POSTCONV_INPUT_MEDIATION"] += "\n冻结验证小子集的 Qwen q/k/v、Falcon x/B/C 单独替换：\n\n" + component_table + "\n单独子量是探索性，不能用最优子量替换预定联合阶段门槛。\n"
    docs["MULTI_STAGE_PIPELINE_MEDIATION"] = "# V34 多阶段流水线\n\n开发和验证中，预定的 RECURRENT_READ 单阶段已在两模型通过。按冻结的条件规则，WRITE_PIPELINE、RECURRENT_PIPELINE、FULL_RECURRENCE_TO_RESIDUAL 不启动；没有测试结果，也没有声称分布式中介。\n"
    docs["MEDIATION_OVERLAP"] = "# V34 中介重叠\n\n只有一个可独立识别的功能阶段通过强门槛。写侧三个阶段均未通过，残差整合与读出为同一可写节点；因此没有两个合格独立阶段可做预定的 S1/S2/S1+S2 重叠分解。REM 值不相加，不给出 serial、redundant 或 complementary 分类。\n"
    control_rows = [(m, fmt(x["matched_restore_median"]), fmt(x["wrong_token_restore_median"]), fmt(x["rec_only_stage_restore_median"]), fmt(x["joint_vs_rec_only_stage_different_fraction"]), fmt(x["KV_read_restore_median_valid"])) for m,x in controls.items()]
    control_table = table(("模型", "匹配恢复", "错 token 恢复", "REC-only 阶段恢复", "Conv 条件差异率", "供体 KV 恢复"), control_rows)
    docs["CONTEXT_SPECIFIC_MEDIATOR"] = "# V34 读出阶段语境特异性\n\n" + control_table + "\n每模型验证集五家族各两状态、同一 Conv 目标；错配值取同状态的另一冻结探针 token。错配控制有 off-manifold 风险，因此只作辅助证据，不独立证明语义匹配。\n"
    docs["CONV_DEPENDENCE"] = "# V34 Conv 依赖\n\n" + control_table + "\n比较 donor REC + donor Conv 与 donor REC + recipient Conv 的读出阶段值，以及它们在同一 Conv-only 目标上的反向插入。阶段哈希差异与恢复差异描述条件性；读出层是宽截面，不能反推出某个局部门控是唯一来源。\n"
    docs["REC_ONLY_MICRO_CONTROL"] = "# V34 REC-only 微阶段对照\n\n" + control_table + "\nREC-only 分支的阶段值并不等同于联合分支，且 REC-only 的高层供体误差见同角色 factorial Parquet。原生阶段值存在差异不等于足够的未来修正；只有精确替换后的响应方向与收益才进入中介结论。\n"
    docs["FUNCTIONAL_MEDIATION_PROFILE"] = "# V34 功能中介画像\n\n" + stage_table + "\nREM/REST 以 REC 在 Conv-only 之上的修正收益为分母；恢复余弦对比插入引出的修正向量与真实 Y11−Y01。只比较无量纲画像，不比较跨模型原始张量范数。\n"
    docs["CROSS_MODEL_MEDIATOR_COMPARISON"] = "# V34 跨模型中介比较\n\n" + stage_table + "\n两模型在独立开发、验证以及最终集均呈读侧通过、写侧未通过的 PROFILE MATCH。该结果支持同一宽功能截面，不证明 Qwen delta 与 Falcon dBx 是同一算法。" + caution + "\n"
    depth_rows = [(m, g, x["positive"], fmt(x["median_removed"]), fmt(x["median_restored"]), fmt(x["median_cosine"])) for m, groups in depth.items() for g,x in groups["groups"].items()]
    docs["DEPTH_LOCALIZATION"] = "# V34 相对深度定位\n\n" + table(("模型", "相对深度组", "正收益状态", "REM 中位", "REST 中位", "余弦中位"), depth_rows) + "\n通过读出阶段后才进行该探索性分析，按每模型 24 个循环层的相对四分位划分，验证集每家族一状态；不把原始层号当跨架构同源层，也不据 5 状态细分精确层位。\n"
    horizon_rows = [(m, g, x["positive"], fmt(x["median_removed"]), fmt(x["median_restored"]), fmt(x["median_cosine"])) for m, groups in horizons.items() for g,x in groups["groups"].items()]
    docs["MULTI_HORIZON_MEDIATION"] = "# V34 多步传播\n\n" + table(("模型", "h", "正收益状态", "REM 中位", "REST 中位", "余弦中位"), horizon_rows) + "\n仅 h1 执行读出补丁，后续 token 的 h2/h4 自然演化；冻结验证子集每家族一状态、每状态用前四个预选探针作为 token 序列。这是探索性小样本传播分析，不替代 h1 正式门槛。\n"
    docs["STRICT_INTERFACE_AUDIT"] = "# V34 严格接口审计\n\n" + mapping + "\n校准原生重放、双向精确写回、未动 recipient KV、同背景/同探针/同位置都由机器记录核查；错误会使整次运行失败。阶段 5 的别名状态是接口限制，不以整个残差向量替代循环贡献。原始哈希和每探针 24 层证据见 interface_audit 与 mediation_audit 文件。\n"
    hashes = table(("模型", "revision", "权重 SHA-256"), [(m, cfg["models"][m]["revision"], ", ".join(cfg["models"][m]["weight_sha256"].values())) for m in ("Q", "F")])
    docs["EXECUTION_MANIFEST"] = "# V34 执行清单\n\n" + hashes + f"\n父提交 `{verify(root)['parent_commit']}`；基础协议 freeze `{verify(root)['freeze_digest']}`；最终开放 freeze `{verify_stage(root,'final_opening')['freeze_digest']}`；裁决 freeze `{verify_stage(root,'adjudication')['freeze_digest']}`。面板：每模型 20 校准、80 开发、40 验证、40 独立最终，跨模型语义配对并排除 V28–V33 正式状态。精确 tensor 原值未入 Git；提交机器 JSON、Parquet、NPZ、哈希及报告。完整文件清单见 `results/v34/processed/v34_integrity_index.json`。\n"
    answers = [
        "1. 高层修正再次复现：是，两模型三角色均通过。",
        "2–5. post-Conv、变换控制、真实更新、循环读出：均可双向精确写入。",
        "6. 残差整合的循环贡献与读出同节点；无法独立识别。",
        "7–10. 各阶段 REM、REST、余弦和 ≥50% 判定见功能画像；只有读出通过。",
        "11–16. 同一读侧截面通过两模型；真实更新、门控、post-Conv 输入及独立残差阶段未通过。",
        "17. 单阶段已通过，预定多阶段流水线无需启动；不声称分布式中介。",
        "18. 两模型无量纲画像同为 read-side dominant。",
        "19–21. 匹配/错配、Conv 条件性、REC-only 与 KV 结果见冻结小子集报告；错配有 off-manifold 限制。",
        "22–23. 相对深度与 h2/h4 是通过阶段后的探索性小子集分析，详见相应报告。",
        "24–28. V34-A/E 成立；B/C/D/F/G/H/I 不成立或未启动，见裁决 JSON。",
        "29. 应用实验获授权，但 V34 未建立实际应用收益。",
        "30. 第三模型复制获授权，但当前结论只覆盖这两个模型。",
    ]
    docs["SCIENTIFIC_ANSWERS"] = "# V34 科学问题逐项回答\n\n" + "\n\n".join(answers) + "\n\n" + caution + "\n"
    docs["COMPLETE_REPORT"] = ("# V34 Complete Report\n\n"
        "**Cross-Model Functional Mediation of REC–Conv Correction**\n\n" + main + " " + caution + "\n\n"
        "## 高层重确认\n\n" + high_table + "\n## 功能阶段双向中介\n\n" + stage_table + "\n"
        + evidence + " Stage 5 is aliased with Stage 4 and is not counted separately. The frozen multi-stage sets were not run because a single stage qualified.\n\n"
        "## 冻结小子集与边界\n\n" + control_table + "\n相对深度和 h2/h4 详见各自报告；它们均为探索性，不改变正式 h1 决策。\n\n"
        + f"最终裁决：`{', '.join(k for k,v in adj['formal_outcomes'].items() if v)}`。应用实验授权={adj['application_experiments_authorized']}，第三模型复制授权={adj['third_model_replication_authorized']}；实际应用收益未建立。"
        + " 机器证据在 `results/v34/processed/`；V1–V33 历史记录未修改。\n")
    report_dir = root / "reports"
    for name in NAMES:
        write(report_dir / f"V34_{name}.md", docs[name])
    combined = "# V34 All Reports\n\n本文件逐一收录 V34 全部 21 份独立报告；各标题下内容与同名文件一致。\n\n"
    for name in NAMES:
        combined += f"\n---\n\n<!-- V34_{name}.md -->\n\n" + docs[name].rstrip() + "\n"
    write(report_dir / "V34_ALL_REPORTS.md", combined)
    cumulative = report_dir / "FINAL_REPORT.md"
    old = cumulative.read_text(encoding="utf-8")
    if "## V34 — Cross-Model Functional Mediation" in old:
        raise RuntimeError("V34 already appended to cumulative final report")
    cumulative.write_text(old.rstrip() + "\n\n## V34 — Cross-Model Functional Mediation\n\n"
        + main + " " + caution + " 独立最终集每模型 40 状态确认；详见 `reports/V34_COMPLETE_REPORT.md` 和 `reports/V34_ALL_REPORTS.md`。\n", encoding="utf-8")
    return {"report_count": len(NAMES)+1, "combined": "reports/V34_ALL_REPORTS.md",
            "cumulative_appended": True}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2, ensure_ascii=False))
