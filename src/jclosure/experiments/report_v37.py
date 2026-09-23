"""Write V37-only reports from sealed records, plus one standalone bundle."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v37 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v37/processed")
REPORTS = Path("reports")
SOURCE = "src/jclosure/experiments/report_v37.py"
MODELS = {"Q": "Qwen3.5-4B", "F": "Falcon-H1-1.5B"}
TOPICS = (
    "FROZEN_STARTING_POINT", "HIGH_LEVEL_RECONFIRMATION", "Q2_Q3_Q4_FRESH_REPLICATION",
    "NATIVE_RECURRENCE_EQUATIONS", "DERIVED_VARIABLE_DEPENDENCY_AUDIT",
    "NATIVE_TERM_DECOMPOSITION", "LOCAL_FIXED_INPUT_FACTORIAL",
    "LOCAL_TO_GLOBAL_REINSERTION", "READ_OPERATOR_DERIVATION",
    "READ_OPERATOR_PREDICTION", "THREE_BRANCH_DESIGN",
    "STATE_OPERATOR_COMPATIBILITY_MATRIX", "GAIN_ROTATION_CONTROLS",
    "OLD_STATE_VS_UPDATE", "TRUE_UPDATE_REINTERPRETATION",
    "PRE_POST_READ_TRANSFORMATION", "UPSTREAM_CONTEXT_DEPENDENCE",
    "Q2_CONTEXT_CONDITIONING", "Q2_DIRECT_INDIRECT_DECOMPOSITION",
    "CROSS_LAYER_OPERATOR_CONDITIONING", "Q2_Q3_Q4_MECHANISTIC_RECONSTRUCTION",
    "STATE_VS_OPERATOR_COUNTERFACTUAL", "OPERATOR_SENSITIVITY_GEOMETRY",
    "TASK_GROUNDED_FUNCTIONAL_TEST", "CROSS_MODEL_OPERATOR_LAW",
    "STRICT_INTERFACE_AUDIT", "EXECUTION_MANIFEST", "SCIENTIFIC_ANSWERS",
    "COMPLETE_REPORT",
)


def _read(root: Path, name: str):
    return json.loads((root / OUT / name).read_text(encoding="utf-8"))


def _optional(root: Path, name: str):
    path = root / OUT / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _fmt(x, digits=3):
    return "—" if x is None else f"{x:.{digits}f}"


def _table(head, rows):
    return "\n".join(["|" + "|".join(head) + "|",
                      "|" + "|".join("---" for _ in head) + "|"] +
                     ["|" + "|".join(str(v) for v in row) + "|" for row in rows])


def _write_once(root: Path, name: str, body: str):
    path = root / REPORTS / name
    value = body.rstrip() + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != value:
        raise RuntimeError(f"V37 report exists with different content: {name}")
    if not path.exists():
        path.write_text(value, encoding="utf-8")
    return path


def run(root: Path):
    cfg = verify(root)["config"]
    for stage in ("sample_pools", "panel", "design", "source_audit",
                  "read_gate_development", "high_level_development",
                  "cross_layer_analysis_development", "local_analysis_development",
                  "read_gate_validation", "high_level_validation",
                  "local_analysis_validation", "final_opening", "adjudication"):
        verify_stage(root, stage)
    for key in MODELS:
        for stage in (f"calibration_{key}", f"operator_geometry_{key}",
                      f"natural_reconstruction_{key}_development",
                      f"native_terms_{key}",
                      f"operator_controls_fit_{key}",
                      f"operator_controls_validation_{key}"):
            verify_stage(root, stage)
        for role in ("development", "validation"):
            verify_stage(root, f"compatibility_{key}_{role}")
    panel = _read(root, "panel_v37.json")
    audit = _read(root, "source_audit_v37.json")
    high = {r: _read(root, f"high_level_{r}_v37.json") for r in ("development", "validation")}
    read = {r: _read(root, f"read_gate_{r}_v37.json") for r in high}
    local = {r: _read(root, f"local_analysis_{r}_v37.json") for r in high}
    cross = _read(root, "cross_layer_analysis_development_v37.json")
    natural = {k: _read(root, f"natural_reconstruction_{k}_development_v37.json") for k in MODELS}
    geometry = {k: _read(root, f"operator_geometry_{k}_v37.json") for k in MODELS}
    terms = {k: _read(root, f"native_terms_{k}_v37.json") for k in MODELS}
    compatibility = {(k,r): _read(root, f"compatibility_{k}_{r}_v37.json")
                     for k in MODELS for r in ("development", "validation")}
    control = {k: _read(root, f"operator_controls_validation_{k}_v37.json") for k in MODELS}
    calibration = {k: _read(root, f"calibration_{k}_v37.json") for k in MODELS}
    opening = _read(root, "final_opening_v37.json")
    adjudication = _read(root, "v37_adjudication.json")
    final = _optional(root, "final_analysis_v37.json")
    if opening["opened"]:
        verify_stage(root, "final_analysis")
        if final is None:
            raise RuntimeError("V37 opened independent final is not complete")
    h_rows, r_rows, l_rows, c_rows, n_rows, g_rows, ctl_rows = [], [], [], [], [], [], []
    term_rows, compatibility_rows = [], []
    for role in high:
        for key, name in MODELS.items():
            h = high[role]["models"][key]
            h_rows.append((role, name, h["states"], _fmt(h["positive_fraction"]),
                           _fmt(h["median_reduction"]), h["families_passing"], h["pass"]))
            r = read[role]["models"][key]
            r_rows.append((role, name, _fmt(r["full_read"]["removed"]),
                           _fmt(r["full_read"]["restored"]),
                           _fmt(r["q2_q3_q4_secondary"]["removed"]),
                           _fmt(r["q2_q3_q4_secondary"]["restored"]),
                           r["q2_q3_q4_secondary"]["families_passing"]))
            x = local[role]["models"][key]
            l_rows.append((role, name, x["states"], _fmt(x["whole"]["cosine"]),
                           _fmt(x["whole"]["rank"]), _fmt(x["whole"]["ordering"]),
                           _fmt(x["median_local_benefit_fraction"]),
                           x["prediction_gate_pass"]))
            cm = compatibility[key, role]
            compatibility_rows.append((role, name,
                                       _fmt(cm["median_B_matched_fraction"]),
                                       _fmt(cm["median_C_matched_fraction"])))
    for key, name in MODELS.items():
        x = cross["models"][key]
        c_rows.append((name, _fmt(x["whole"]["median_mixer_change_cosine"]),
                       _fmt(x["whole"]["median_future_order_accuracy"]),
                       sum(v["pass"] for v in x["families"].values()),
                       x["cross_layer_gate_pass"]))
        x = natural[key]
        n_rows.append((name, _fmt(x["median_Q234_fraction_of_full"]),
                       _fmt(x["median_Q234_direction_cosine"]),
                       _fmt(x["median_natural_Q2_enablement"])))
        x = geometry[key]
        g_rows.append((name, x["rows"], _fmt(x["median_operator_cosine_to_A"]),
                       _fmt(x["median_operator_norm_ratio_to_A"])))
        t = terms[key]
        term_rows.append((name, t["rows"], _fmt(t["median_old_read_difference_norm"]),
                          _fmt(t["median_update_read_difference_norm"]),
                          _fmt(t["fraction_update_independent_old_state"])))
        x = control[key]["by_method"]
        for stage in ("raw", "mixer"):
            for method in ("native", "gain", "fixed_rotation"):
                m = x[f"{stage}_{method}"]
                ctl_rows.append((name, stage, method, _fmt(m["median_cosine"]),
                                 _fmt(m["median_relative_vector_error"]),
                                 _fmt(m["median_relative_magnitude_error"])))
    htable = _table(("角色", "模型", "状态", "正效应", "误差减少", "家族", "通过"), h_rows)
    rtable = _table(("角色", "模型", "全读移除", "全读恢复", "Q234移除", "Q234恢复", "Q234家族"), r_rows)
    ltable = _table(("角色", "模型", "状态", "局部余弦", "幅度秩相关", "成对排序", "单层下游收益比例", "局部门槛"), l_rows)
    ctable = _table(("模型", "局部余弦", "下游排序", "通过家族", "跨层门槛"), c_rows)
    ntable = _table(("模型", "Q234自然恢复比例", "方向余弦", "Q2条件增强"), n_rows)
    gtable = _table(("模型", "校准算子行", "相对A算子余弦", "相对A范数"), g_rows)
    ctl_table = _table(("模型", "阶段", "解释", "方向余弦", "相对向量误差", "相对幅度误差"), ctl_rows)
    term_table = _table(("模型", "校准行", "旧状态读出差范数", "更新读出差范数", "更新独立比例"), term_rows)
    compatibility_table = _table(("角色", "模型", "B对角最强比例", "C对角最强比例"), compatibility_rows)
    final_text = (f"已开放 `{opening['selected_class']}`；独立最终两模型局部读算子门槛"
                  f"{'通过' if final['both_models_local_read_law_pass'] else '未通过'}，详见 `final_analysis_v37.json`。"
                  if opening["opened"] and final is not None else
                  "独立最终未开放或尚未完成；不能报告最终泛化结论。")
    final_table = (_table(("模型", "状态", "raw余弦", "raw排序", "mixer余弦", "mixer排序", "门槛"),
                          [(MODELS[k], final["models"][k]["states"],
                            _fmt(final["models"][k]["whole"]["raw_cosine"]),
                            _fmt(final["models"][k]["whole"]["raw_ordering"]),
                            _fmt(final["models"][k]["whole"]["mixer_cosine"]),
                            _fmt(final["models"][k]["whole"]["mixer_ordering"]),
                            final["models"][k]["local_read_law_pass"])
                           for k in MODELS]) if final is not None else "独立最终未观察。")
    reports = {}
    def put(topic, title, body):
        reports[f"V37_{topic}.md"] = f"# V37 {title}\n\n{body.strip()}\n"
    put("FROZEN_STARTING_POINT", "冻结起点", f"""
V37 父提交 `{cfg['parent_commit']}` 为已推送 V36；V1–V36 正式文件和结果完全不变，V36 正式观测不计入 V37。仅复用经新校准审计的工程 instrumentation/方程。

新样本池：校准 `{len(panel['calibration'])}`、开发 `{len(panel['development'])}`、正式验证 `{len(panel['validation'])}`、独立最终 `{len(panel['independent_final'])}` states/model。验证和最终各有独立预封存池；V18 training-source 状态不进入 V37 formal validation/final。五任务族各为 4/16/8/8 states。池封存时两模型响应均未观测。样本为同任务模板的新程序/提示，不宣称任务族分布外泛化。

样本分组哈希：`{json.dumps(panel['role_hashes'], sort_keys=True, ensure_ascii=False)}`。源审计：`{sha256_file(root/OUT/'source_audit_v37.json')}`。""")
    put("HIGH_LEVEL_RECONFIRMATION", "高层条件效应重新验证", f"{htable}\n\n六 probe 先在同一状态内合并，state 是 bootstrap 独立单位；验证池绝非 V18 剩余 training-source。")
    put("Q2_Q3_Q4_FRESH_REPLICATION", "Q2/Q3/Q4 新样本复核", f"{rtable}\n\n完整读出移除/恢复属于接口检验；Q2+Q3+Q4 是单独标记的次级复核，不重开 V35/V36 的旧最终。")
    put("NATIVE_RECURRENCE_EQUATIONS", "原生递推方程与等价校准", f"""
Qwen 单 token 实数形式：`S_d=G S; delta=β(v-kᵀ S_d); S'=S_d+k⊗delta; r=qᵀS'`。Falcon：`S'=dA⊙S+dBx; r=CᵀS'+Dx`。实际比较采用安装源码的 dtype/门控/归一化/输出投影顺序，不把代数式当成 BF16 逐位等式。

V37 新校准：Q/F 每模型 `{calibration['Q']['states']}`/`{calibration['F']['states']}` 个状态，各 `{calibration['Q']['equality_rows']}`/`{calibration['F']['equality_rows']}` 个 state×site×cell×stage 等价审计行；最大绝对误差 `{calibration['Q']['max_abs_error']}`/`{calibration['F']['max_abs_error']}`，instrumented logits/cache/endpoints 均 bitwise。V36 旧校准没有替代此步。""")
    put("DERIVED_VARIABLE_DEPENDENCY_AUDIT", "派生变量依赖审计", """
Qwen：`(h,Conv)→q,k,v`，`h→G,β,z`，`(S,G,k)→m=kᵀGS`，`(v,m,β)→delta`，`(S,delta,k)→S'→r→norm→mixer`。`delta/TRUE_UPDATE` 依赖旧 S，不是独立于旧状态的新写入。

Falcon：`(h,Conv)→x,B,C`，`h→dt,dA,gate`，`(dt,B,x)→dBx`，`(S,dA,dBx)→S'→r→norm→mixer`。固定 h/Conv 时 dBx 独立于旧 S。缓存写入和输出均由原生重算，不把历史派生张量独立拼接。""")
    put("NATIVE_TERM_DECOMPOSITION", "原生项分解", f"""
Qwen 固定输入下 `r=qᵀG S - qᵀ[k⊗β(kᵀGS)] + qᵀ[k⊗βv]`：旧状态保持项、旧状态依赖的移除项、当前输入写入项。Falcon 固定输入下 `r=Cᵀ(dA⊙S)+CᵀdBx+Dx`。

{term_table}

这些是 V37 新校准池的依赖一致代数项，不是三项被独立操控后的行为因果结论。V37 未对每项逐项做完整双向整模型干预，因此旧/更新的正式优势类别不确认。""")
    put("LOCAL_FIXED_INPUT_FACTORIAL", "固定输入 3×3 因子实验", f"{ltable}\n\nA/B/C 为同前缀的三个自然 token，固定 Conv-only 轨迹的 h，逐格原生重算 `S_i×Conv_j`，不复制 donor mixer。逐 probe 预测先封存，后执行九格并核对请求/实现哈希。表中单层下游收益比例与局部代数门槛分列，不能互换。")
    put("LOCAL_TO_GLOBAL_REINSERTION", "局部输出自然下游重插入", f"{ltable}\n\n四格 AA/BA/AB/BB 的本地重算 mixer 在指定单层插入 Conv-only 支路，后层自然传播；其恢复比例只是单层效应，未将其加总成全深度机制。分母绝对收益低于 1.0 时比例缺失，只报告绝对量。")
    put("READ_OPERATOR_DERIVATION", "状态读算子推导", """
固定 h/Conv，Qwen 的实数线性状态读映射为 `qᵀ[G I - βk(kᵀG)]S`；Falcon 为 `Cᵀ(dA⊙S)`。两者均是局部状态差对 raw read 的映射。V37 用真实有限精度方程分别算 3×3 预测，再与原生运行比较；归一化/门控和 output projection 另作后读阶段，不直接等同 raw 线性算子。""")
    put("READ_OPERATOR_PREDICTION", "前瞻读算子预测", f"{ltable}\n\n{final_table}\n\n开发、验证和获准后的独立最终预测分别先冻结，之后才观察对应 3×3 局部 outcome。门槛为余弦≥0.90、幅度秩相关≥0.70、成对排序≥0.80、≥4/5 家族且两模型各自通过；这是局部计算层级，不自动证实跨层功能。")
    put("THREE_BRANCH_DESIGN", "三个自然 token 分支", "A=recipient、B=donor、C=同前缀第三自然 token。A/B/C 均来自预冻结、响应盲的 token triplet；各自的 REC 与 Conv 缓存形成九格。C 是自然同前缀错配对照，不是随机合成 activation；跨家族和随机错配不用于主要语义判定。")
    put("STATE_OPERATOR_COMPATIBILITY_MATRIX", "状态×算子兼容矩阵", f"九格局部 raw/mixer 预测和观测向量保存于分模型、分角色 NPZ（大张量仅保留工作区，Git 只记录哈希）。逐格自然缓存和局部输入 hash 在 Parquet。\n\n{compatibility_table}\n\n对角最强是次级、描述性模式，不是主公式预测门槛；全局未来只对预先指定四格重插入，未对九格都执行未来因果比较。")
    put("GAIN_ROTATION_CONTROLS", "标量增益与冻结旋转对照", f"{ctl_table}\n\n标量增益与 rank-16 固定正交旋转均只在 development 的局部状态差上拟合，按模型×层位×阶段×目标算子冻结；validation 不重拟合。比较原生预测、增益和旋转的方向/向量误差/幅度误差。若简单对照接近原生，不能宣称复杂算子几何是独特解释。")
    put("OLD_STATE_VS_UPDATE", "旧状态与当前更新", "Qwen TRUE_UPDATE/`delta` 依赖旧 S，因此将其与旧状态从不同自然来源拼接并不构成干净的 OLD_ONLY/UPDATE_ONLY。Falcon 固定 h/Conv 的 dBx 独立于旧 S，但 V37 未完整执行双向 OLD/UPDATE 因子未来端点；不判定 OLD_STATE_DOMINANT 或 CURRENT_UPDATE_DOMINANT。")
    put("TRUE_UPDATE_REINTERPRETATION", "V34 TRUE_UPDATE 边界重释", "V34 旧结果保持原样。Qwen 中移植派生 TRUE_UPDATE 会带入另一旧状态计算的 m，属依赖错配；Falcon 的 dBx 不依赖旧状态，但单独读出仍可能受 C/门控/后层影响。V37 未通过新双向干预在 A–E 解释间裁决，不能把 V34 失败概括成当前更新无作用。")
    put("PRE_POST_READ_TRANSFORMATION", "读前与读后转换", "V37 在固定输入九格记录 raw、normalized、mixer 的交互范数、方向和自然轨迹对比，并重插入 mixer 检查下游。单凭读后交互放大或旋转不足以判定该阶段中介；没有独立、双向的 norm/gate-only 整模型干预，故 POST_READ_TRANSFORMATION 不确认。")
    put("UPSTREAM_CONTEXT_DEPENDENCE", "上游上下文依赖", f"{ctable}\n\nQ2 输出改变后 Q3/Q4 因子和 hidden input，局部映射方向吻合，但未来排序未过冻结门槛。这是上游影响的证据，不等于上游变化足以解释跨层行为。")
    put("Q2_CONTEXT_CONDITIONING", "Q2 条件化实验", f"{ctable}\n\n只替换上游 Q2 读输出，后层状态读/网络自然执行；在后层目标 REC 状态干预前冻结预测。开发集下游排序未达到 ≥0.75、≥4/5 家族，故该候选未进入正式验证。")
    put("Q2_DIRECT_INDIRECT_DECOMPOSITION", "Q2 直接/间接路径", "已观察 Q2 介入对后层隐藏输入与读因子的影响，但未构造可分离的 Q2 直接和 Q3/Q4 间接路径双向因子干预。现有对比无法识别路径特异中介量，不报告 DIRECT_DOMINANT 或 INDIRECT_DOMINANT。")
    put("CROSS_LAYER_OPERATOR_CONDITIONING", "跨层算子条件化", f"{ctable}\n\n局部 mixer 变化余弦约 1 是原生公式局部等价；真正的前瞻六 probe 下游排序没有过门槛。验证/最终不为失败的跨层候选打开，结论为开发集未证实，而非跨层作用不存在。")
    put("Q2_Q3_Q4_MECHANISTIC_RECONSTRUCTION", "Q2/Q3/Q4 自然重建", f"{ntable}\n\n只替换指定段的 REC 状态，后层自然传播，不复制 Q3/Q4 或任意未来输出。恢复比例和方向说明组合仍有功能效应，却不单独定位是哪一条算子条件化路径。")
    put("STATE_VS_OPERATOR_COUNTERFACTUAL", "状态与算子反事实", "固定同一 h 的 A/B/C REC × A/B/C Conv 九格是主要同前缀天然反事实。下游四格亦自然重插入；不能把错配格的非自然组合直接解释为模型平时会采取的行为。独立 final 若开放，只检验已选的局部读算子类。")
    put("OPERATOR_SENSITIVITY_GEOMETRY", "算子敏感性几何", f"{gtable}\n\n奇异谱、token 分支、site 和 family 标注在 calibration-only Parquet/NPZ。这里是 exact raw-state 线性映射的解析几何；没有前瞻因果低秩压缩检验，故不称低秩功能机制。")
    put("TASK_GROUNDED_FUNCTIONAL_TEST", "任务语义功能测试", "未完成外部定义的五类语义变量恢复实验；向量 donor 误差/读出恢复不等于正确答案、绑定值、图节点或状态转移变量被恢复。V37 最多确认已测的局部计算和接口层级，不宣称 LEVEL 4 TASK FUNCTION。")
    put("CROSS_MODEL_OPERATOR_LAW", "跨模型算子法则", f"{ltable}\n\nQwen 与 Falcon 分别用自身原生方程实现同一个抽象：固定输入下旧状态经 Conv 条件化读映射；微观公式不同。共同局部门槛与跨层门槛分开报告，不能把两模型局部等价改写为共享全局修正算法。")
    put("STRICT_INTERFACE_AUDIT", "严格接口与阴性对照", "V37 自己的校准对两模型各 20 states×6 sites×9 cells×3 stages，最大误差零，instrumentation bitwise。记录自然 Conv-only、REC-only、joint、全读出移除/恢复、同前缀 A/B/C 缓存错配，以及 development-only 增益/旋转。随机打乱、跨家族错配、独立 post-read 双向移除未覆盖，报告中不得视作已通过。")
    put("EXECUTION_MANIFEST", "执行与完整性清单", f"""
父提交 `{cfg['parent_commit']}`；V37 基础封存 `{sha256_file(root/'artifacts/computational_origin_v37.freeze.json')}`。四池 20/80/40/40 states/model，validation 和 independent final 都是预封存新池；V18 train-source 不进入 formal roles。模型权重/配置/tokenizer hash 与安装源码审计在 `artifacts/computational_origin_v37.freeze.json`、`source_audit_v37.json`，每一步的输入 SHA 和结果 SHA 在独立 stage freeze。独立最终开放 `{opening['opened']}`，候选 `{opening['selected_class']}`；V36 历史未改动。""")
    put("SCIENTIFIC_ANSWERS", "科学问题答复", f"""
1. 新 V37 高层条件效应：开发和正式验证见高层表。2. Q2/Q3/Q4 次级读接口：见 `{rtable.splitlines()[-1]}`。3. 两架构原生公式及依赖：见本版方程/依赖报告。4. 局部状态×算子效应：{ltable}。

5. 简单增益/固定旋转能否解释：见增益旋转对照表，结论限定在已测局部向量。6. Q2 条件化是否建立跨层机制：否，开发下游排序未过门槛。7. 全段自然重建：{ntable}。8. OLD/UPDATE、读后中介及任务语义：缺少对应双向/外部变量实验，不作正式确认。9. 独立最终：{final_text}。10. V36 结果完全排除，未借用旧正式样本或旧结论。""")
    conclusion = (f"最高支持层级为 `{adjudication['strongest_supported_level']}`；"
                  + ("局部读算子类通过开发和独立正式验证；" if opening["opened"] else
                     "没有机制类同时通过开发与正式验证；"))
    put("COMPLETE_REPORT", "Complete Report — Computational Origin of REC–Conv Conditional Effects", f"""
结论：V37 以新独立样本重新确认高层 REC–Conv 条件效应和读出接口。{conclusion}跨层 Q2→后层下游排序的开发判据失败；局部原生计算等价不得提升为跨层或任务语义机制。{final_text}

## 样本与先验封存

校准/开发/正式验证/独立最终为 20/80/40/40 states/model、五家族 4/16/8/8。正式验证池和独立最终池彼此独立、在正式 intervention outcome 前封存。V18 剩余 training-source 与 V36 正式结果均不计入 V37。

## 高层与接口

{htable}

{rtable}

## 局部读算子与下游

{ltable}

{final_table}

{ctable}

{ntable}

## 对照和解释上界

{ctl_table}

两模型新校准各 3240 等价行，最大误差零；但 source-derived 公式对 source-derived 运行精确属于 LEVEL 2 计算核验。单层自然下游恢复、Q2 条件化和语义任务问题分别单独判定，不由局部余弦代替。旧状态/更新拆分和 post-read 双向中介未完整执行，结论保持未判定。
""")
    for topic in TOPICS:
        name = f"V37_{topic}.md"
        if name not in reports:
            raise RuntimeError(f"Missing V37 topic {name}")
        _write_once(root, name, reports[name])
    cumulative = root / REPORTS / "FINAL_REPORT.md"
    existing = cumulative.read_text(encoding="utf-8")
    if "<!-- V37_START -->" not in existing:
        addition = ("\n\n<!-- V37_START -->\n"
                    "## V37 — Computational Origin of REC–Conv Conditional Effects\n\n"
                    + reports["V37_COMPLETE_REPORT.md"].split("\n\n", 1)[1].split("\n\n##", 1)[0]
                    + "\n\n完整专题和机器记录：`reports/V37_COMPLETE_REPORT.md`、"
                      "`reports/V37_ALL_REPORTS.md`、`results/v37/processed/v37_integrity_index.json`。\n"
                      "<!-- V37_END -->\n")
        cumulative.write_text(existing.rstrip() + addition, encoding="utf-8")
    elif "<!-- V37_END -->" not in existing:
        raise RuntimeError("Incomplete existing V37 cumulative section")
    paths = sorted(root / REPORTS / f"V37_{topic}.md" for topic in TOPICS)
    bundle = ["# V37 All Reports — Computational Origin of REC–Conv Conditional Effects", "",
              "本文件独立收录 V37 全部专题报告；V36 不在此计分。", "",
              "## 文件 SHA-256", "",
              _table(("报告", "SHA-256"), [(p.name, sha256_file(p)) for p in paths])]
    for path in paths:
        bundle.extend(["", "---", "", f"## {path.name}", "",
                       path.read_text(encoding="utf-8").rstrip()])
    all_path = _write_once(root, "V37_ALL_REPORTS.md", "\n".join(bundle))
    result_paths = sorted(p for p in (root / OUT).iterdir()
                          if p.is_file() and p.name != "v37_integrity_index.json")
    report_paths = sorted((root / REPORTS).glob("V37_*.md"))
    index = {"version": "V37", "parent_commit": cfg["parent_commit"],
             "base_freeze_sha256": sha256_file(root/"artifacts/computational_origin_v37.freeze.json"),
             "final_opened": opening["opened"], "finalist": opening["selected_class"],
             "results": {str(p.relative_to(root)): sha256_file(p) for p in result_paths},
             "reports": {str(p.relative_to(root)): sha256_file(p) for p in report_paths},
             "cumulative_report_sha256": sha256_file(cumulative),
             "historical_v36_modified": False}
    index_path = root / OUT / "v37_integrity_index.json"
    write_json_atomic(index_path, index)
    seal = stage_freeze(root, "report",
                        [SOURCE, str(index_path.relative_to(root)),
                         *[str(p.relative_to(root)) for p in report_paths]],
                        {"report_count": len(report_paths), "result_count": len(result_paths),
                         "index_sha256": sha256_file(index_path), "final_opened": opening["opened"]})
    return {"freeze_digest": seal["freeze_digest"], "reports": len(report_paths),
            "results": len(result_paths), "all_reports": str(all_path.relative_to(root))}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2, ensure_ascii=False))
