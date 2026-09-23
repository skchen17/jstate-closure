"""Generate V36 reports only from sealed machine records; preserve all earlier reports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.protocol_v36 import stage_freeze, verify, verify_stage

OUT = Path("results/v36/processed")
REPORTS = Path("reports")
SOURCE = "src/jclosure/experiments/report_v36.py"
MODEL = {"Q": "Qwen3.5-4B", "F": "Falcon-H1-1.5B"}
ROLES = ("development", "validation")


def _read(root, name):
    return json.loads((root / OUT / name).read_text())


def _table(headers, rows):
    return "\n".join(["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"] +
                     ["|" + "|".join(str(v) for v in row) + "|" for row in rows])


def _f(x, digits=3):
    return "—" if x is None else f"{x:.{digits}f}"


def _yes(x):
    return "通过" if x else "未通过"


def _write_once(root, name, body):
    path = root / REPORTS / name
    text = body.rstrip() + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"V36 append-only report already exists with different content: {name}")
    else:
        path.write_text(text, encoding="utf-8")
    return path


def run(root: Path):
    cfg = verify(root)["config"]
    for stage in ("v35_audit", "panel", "design", "high_level_development", "high_level_validation",
                  "local_analysis_development", "local_analysis_validation", "final_opening", "adjudication"):
        verify_stage(root, stage)
    for key in MODEL:
        for stage in (f"interface_{key}", f"equation_{key}"):
            verify_stage(root, stage)
        for role in ROLES:
            for stage in (f"factorial_{key}_{role}", f"local_{key}_{role}",
                          f"controls_{key}_{role}", f"control_analysis_{key}_{role}",
                          f"factor_hashes_{key}_{role}"):
                verify_stage(root, stage)
    audit = _read(root, "v35_audit_v36.json")
    high = {r: _read(root, f"high_level_{r}_v36.json") for r in ROLES}
    local = {r: _read(root, f"local_analysis_{r}_v36.json") for r in ROLES}
    controls = {(k, r): _read(root, f"control_analysis_{k}_{r}_v36.json") for k in MODEL for r in ROLES}
    equation = {k: _read(root, f"equation_audit_{k}_v36.json") for k in MODEL}
    opening = _read(root, "final_opening_v36.json")
    adjud = _read(root, "v36_adjudication.json")
    if opening["opened"]:
        raise RuntimeError("V36 report generator currently only supports the sealed-negative path")
    high_rows = []
    local_rows = []
    control_rows = []
    for role in ROLES:
        for key in MODEL:
            h = high[role]["models"][key]
            x = local[role]["models"][key]
            c = controls[key, role]
            high_rows.append((role, MODEL[key], h["rows"], _f(h["positive_fraction"]),
                              _f(h["median_reduction"]), _f(h["median_alignment"]),
                              h["families_passing"], _yes(h["pass"])))
            local_rows.append((role, MODEL[key], x["state_count"],
                               _f(x["whole"]["median_predicted_observed_cosine"]),
                               _f(x["whole"]["median_magnitude_rank_spearman"]),
                               _f(x["whole"]["median_matched_superiority_fraction"]),
                               _f(x["whole"]["median_local_downstream_fraction"]),
                               _yes(x["same_class_operator_and_causal_pass"])))
            control_rows.append((role, MODEL[key], _f(c["median_upper_fraction"]),
                                 _f(c["median_old_projection"]),
                                 _f(c["median_dependent_update_projection"]),
                                 _f(c["median_shuffled_cosine"]),
                                 _f(c["median_sign_flip_opposite_cosine"])))
    high_table = _table(("角色", "模型", "状态", "正收益率", "中位修正减少", "残差对齐", "家族", "门槛"), high_rows)
    local_table = _table(("角色", "模型", "状态", "局部公式余弦", "幅度排序", "匹配优势率", "下游恢复率", "联合门槛"), local_rows)
    control_table = _table(("角色", "模型", "供体混合器上界恢复率", "旧状态投影", "状态依赖更新投影", "打乱算子余弦", "反号对向余弦"), control_rows)
    audit_outcome = audit["audit_outcome"]
    outcome = "V36-A/B/H 未确认；独立最终封存。V36-C 与 D–G 缺少预定的充分因果门槛，不能正式确认；V36-I=true 仅作所测四层局部解释不足的有界结论。"
    reports = {}
    reports["V36_NATIVE_RECURRENCE_EQUATIONS.md"] = f"""# V36 原生一 token 递推方程

以实际安装的 Transformers 源码和已冻结的 V34 重放钩子为准：Qwen 文件 SHA-256 `{cfg['native_equations']['source_sha256']['Q']}`，Falcon 文件 SHA-256 `{cfg['native_equations']['source_sha256']['F']}`。下列是单 token 缓存解码，不外推到预填充/分块路径。

## Qwen3.5-4B

`h:[1,1,2560]` 经 `in_proj_qkv` 与深度可分离 Conv 得 `q,k,v`；查询/键从 16 个 key head 复制到 32 个 value head，按末维 L2 归一化，转置到 `[batch,head,time,dim]`，转 fp32，`q` 再乘 `1/sqrt(128)`。原生缓存 `S:[1,32,128,128]`。`g=-exp(A_log)*softplus(a+dt_bias)`，`beta=sigmoid(b)`，两个控制量来自同一个局部输入 `h`，而 `q,k,v` 同时依赖 Conv 缓存与 `h`。

`G=exp(g)`；`S_d=G*S`；`m=sum_key(S_d*k)`；`delta=beta*(v-m)`；`W=k ⊗ delta`；`S'=S_d+W`；`r_raw=sum_key(S'*q)`。核心读出先转回查询原 dtype（BF16），再逐 value head 用 `RMSNormGated(r_raw,z)`，`z=in_proj_z(h)`，然后拼接 32 heads 并经 `out_proj` 得 mixer output `[1,1,2560]`。线性注意力 decoder layer 把此 mixer output 直接加到 residual，然后经过后层归一化与 MLP。

实数代数下，固定局部输入/Conv 时 `Δr_raw=qᵀ[GΔS-k⊗(beta*kᵀGΔS)]`。执行时还须遵守 BF16/fp32 转换，不能把 BF16 差分再次舍入误称精确线性。

## Falcon-H1-1.5B

`h:[1,1,2048]` 经输入倍率、`in_proj` 与 `mup_vector` 分为 gate、Conv 输入、dt。Conv 缓存滚动后深度可分离卷积与激活产生 `x,B,C`；`dt=clamp(softplus(dt+dt_bias))`，`A=-exp(A_log)`，`dA=exp(dt*A)`，`dBx=dt*B*x`。原生缓存 `S:[1,48,64,256]`，head 48、head_dim 64、group 1、state_dim 256，`B/C` 按 group→head 扩展。`S'=S*dA+dBx`；`r_raw=CᵀS'+D*x`，其中 D 是 direct/skip。随后 `mamba_rms_norm=true`，执行带 gate 的 `norm(r_raw,gate)`，`out_proj` 产生 mixer output `[1,1,2048]`；decoder layer 再乘 `ssm_out_multiplier` 才加入 residual（同时还有 attention 支路）。

固定局部输入/Conv 时，实数代数 `Δr_raw=Cᵀ(dA⊙ΔS)`；实际有限精度用两次原生公式计算后在 fp32 做差。

## 实现等价校准

{_table(('模型','校准状态','状态×层×REC 比较','原始读出最大绝对差','原生完整重放'), [(MODEL[k], equation[k]['states'], equation[k]['rows'], _f(equation[k]['max_error'],6), 'bitwise 通过') for k in MODEL])}

每模型 20 个未用于开发/验证的校准状态，四个预定层，每层 recipient/donor 两种旧状态，共 160 次。原生完整 mixer/缓存/输出重放在 `interface_*.json`；逐原始读出对照在 `equation_audit_*.parquet`。这是实现等价检验，不是未来行为机制证明。
"""
    reports["V36_DERIVED_VARIABLE_DEPENDENCY_AUDIT.md"] = """# V36 派生变量依赖审计

## Qwen

`h → (a,b,z)`；`(h,Conv_cache) → (q,k,v)`；`a → g → G`；`b → beta`；`(S,G) → S_d`；`(S_d,k) → m`；`(v,m,beta) → delta`；`(k,delta) → W`；`(S_d,W) → S'`；`(S',q) → r_raw`；`(r_raw,z) → gated_norm → out_proj → mixer → residual`。

因此 `S_d,m,delta,W,S',r_raw` 都依赖旧 REC 状态。V34 的 TRUE_UPDATE `W` 并不是与旧状态独立的新写入；移植 `W_B` 而保留 `S_A` 是可执行的张量干预，却不是依赖一致的“仅写入”反事实。独立于旧状态的当步写入项是 `k⊗beta*v`，状态依赖的移除项是 `-k⊗beta*(kᵀG S)`，必须明确拆开。

## Falcon

`h → (gate,dt,conv_input)`；`(conv_input,Conv_cache) → (x,B,C)`；`dt → dA`；`(dt,B,x) → dBx`；`(S,dA) → old_decay`；`(old_decay,dBx) → S'`；`(S',C,x,D) → r_raw`；`(r_raw,gate) → gated_norm → out_proj → mixer → ssm_out_multiplier → residual`。

固定 h/Conv 时，`dBx` 与旧 `S` 独立，REC 状态差只经 `S*dA` 入读出。任何改变 S 的干预均重算其下游 S'、raw read 与归一化输出；未把历史派生张量伪装成独立变量。两模型角色对应而微观方程不同。
"""
    reports["V36_HIGH_LEVEL_RECONFIRMATION.md"] = f"""# V36 新面板高层重确认

{high_table}

同一状态的六个 probe 先合并为一个五块标准化向量，再用状态为独立 bootstrap 单位。表中正收益、减少、对齐和家族门槛均来自 `high_level_*_v36.json` 与 factorial Parquet。REC-only 弱于 joint、Conv-only 已有较大供体方向贡献；这里只可称条件 REC 效应，不称模型内部显式误差向量。两模型开发/验证都获准进入局部机制实验；独立最终仍受另外的机制门槛约束。
"""
    reports["V36_LOCAL_FIXED_INPUT_FACTORIAL.md"] = f"""# V36 同输入局部 REC×Conv 因子实验

每个模型仅用冻结的相对层位 2/8/14/20（Qwen 实际层 `{json.loads((root/OUT/'design_Q_v36.json').read_text())['local_layers']}`；Falcon `{json.loads((root/OUT/'design_F_v36.json').read_text())['local_layers']}`）。在 Conv-only 自然轨迹捕获同一个 mixer 输入 h；局部构造 `S_A,S_B,S_C × Conv_A,Conv_B,Conv_C` 九格，主要四格为 AA/BA/AB/BB。每个格子以原生 mixer/缓存重算，而不是直接复制 donor 输出。原始、归一化、mixer 与自然轨迹交互均保存于逐 probe Parquet；局部输入、REC/Conv 请求与实现哈希保存于 proof Parquet。

{local_table}

三条件方向/幅度公式预测为真，但这个代数等价本身不说明供体匹配或下游充分性。四层没有按验证结果适应性重选。
"""
    reports["V36_LOCAL_TO_GLOBAL_REINSERTION.md"] = f"""# V36 局部结果的整模型重插入

在 Conv-only 全模型支路的单一预定层，用同输入重算的 `BB` mixer output 或单独的交互增量替换该层输出，然后让剩余层自然传播到同一六 probe 终点。未复制任何下游 donor 输出。每状态四层结果在统计前聚合；自然 REC 收益低于冻结绝对分母 1.0 的状态不计算不稳定比例。

{local_table}

冻结要求状态中位局部恢复率 ≥0.20 且 ≥4/5 家族；开发与验证两模型均未通过。单层局部作用存在，却不足以重建整体 REC 条件收益。供体 mixer 输出复制只作为上界对照，见严格接口审计。
"""
    reports["V36_READ_OPERATOR_DERIVATION.md"] = """# V36 原生 REC 读算子推导

在固定 h 和 Conv 后，Qwen 的 `M_Q(u)` 是 `qᵀ[G· - k⊗(beta*kᵀG·)]`；Falcon 的 `M_F(u)` 是 `Cᵀ(dA⊙·)`。这里 u 含实际 post-Conv q/k/v 或 x/B/C 与来自 h 的门控/衰减；不是仅凭层名推测读写。当前输入驱动项（Qwen `k⊗beta*v`、Falcon `dBx`）在 REC 差分中消去，Falcon direct/skip `D*x` 也消去。归一化/门控与输出投影是读后另一个非线性阶段，不能简单并入线性 M。

预测在原始读出处用同一源代码顺序、dtype 转换计算 `raw(S_B,u)-raw(S_A,u)`，先分别 BF16→fp32，再相减。将两个 BF16 读出先相减会额外舍入，曾在无正式记录的调试运行中被检测并修正；正式记录采用前者。源代码 SHA 与逐状态等价校准见原生方程报告。
"""
    reports["V36_MATCHED_OPERATOR_PREDICTION.md"] = f"""# V36 供体状态与 Conv 读算子匹配预测

固定 `ΔS=S_B-S_A`，在自然 A、匹配 B、第三自然 token C 的 Conv 条件下，先用原生方程预测原始读出差，再运行本地 mixer。状态级中位数如下；六 probe×四层不作为独立样本。

{local_table}

方向余弦和三条件幅度排序达到 1.0，说明源代码公式被忠实复现；但“B 的状态效应大于 A 与 C”的预定匹配优势率远低于 0.80，且四层重插入 <0.20。五家族没有一类达到联合门槛。不能把精确代数等式误当成匹配假说被证实。
"""
    reports["V36_THREE_WAY_COMPATIBILITY_MATRIX.md"] = """# V36 三自然 token 的 3×3 兼容性矩阵

局部固定输入阶段确实逐 probe 运行了 `S_i × Conv_j` 九格（i,j=A/B/C），每格请求/实现的 REC、Conv 哈希在 `local_proof_*.parquet`。其中 A/B 的原始读出、归一化读出与 mixer 交互进入主记录；C 格用于冻结的第三 token 算子预测。各九格完整输出矩阵没有独立落盘，故**不报告**任何未保存的逐格数值或“对角更优”结论。

由于匹配与局部下游门槛均未过，未来响应的九格矩阵按预先规则未获授权。后续若要检验对角兼容性，须新实验先冻结非循环局部/未来评分，而不能从现有 donor 误差直接定义匹配再用同一误差评估。
"""
    reports["V36_OLD_STATE_VS_UPDATE.md"] = f"""# V36 旧状态与当步更新

Qwen：状态差同时改变 `G*S` 与经 `kᵀG*S` 计算的移除/TRUE_UPDATE；不应把 TRUE_UPDATE 的差称为独立新写入。Falcon：固定 h/Conv 的 `dBx` 与 S 独立，REC 差在原始读出只经旧状态衰减支路。下表是基于原生项计算的原始读出投影及供体 mixer 输出上界，并非每项独立下游因果充分性证明。

{control_table}

OLD_ONLY、UPDATE_ONLY 的严格依赖一致构造在固定 h/Conv 下分别退化为完整 REC 差与零 REC 差；若用 donor 状态派生的更新搭配 recipient 旧状态，那是混合反事实，须单独标为干预，不能当纯写入。V36-D/E/F 不凭张量范数正式确认。
"""
    reports["V36_TRUE_UPDATE_REINTERPRETATION.md"] = """# V34 TRUE_UPDATE 失败的 V36 重新解释

Qwen 的 `W=k⊗beta*(v-kᵀG S)` 依赖旧 S。只替换 W 而不替换 `G S`，既漏掉旧状态项，也让 donor 派生的移除项与 recipient 旧状态不一致；所以 V34 TRUE_UPDATE 单独失败不能证明更新无关。Falcon 的 `dBx=dt*B*x` 固定输入时与 S 无关，TRUE_UPDATE 交换没有携带 REC 差；REC 的条件作用通过 `S*dA` 被 C 读取。两种架构原因不同。

本版在校准与开发/验证的公式重算支持上述依赖关系；没有对 V34 全深度 TRUE_UPDATE 历史数据作追认或覆写，也未声称重跑了一个全深度更新干预。
"""
    reports["V36_PRE_POST_NORMALIZATION.md"] = f"""# V36 读出归一化前后

{_table(('角色','模型','局部 raw→norm 交互幅度中位比','旧状态投影','状态依赖更新投影'), [(r,MODEL[k],_f(local[r]['models'][k]['normalization_gain_median']),_f(controls[k,r]['median_old_projection']),_f(controls[k,r]['median_dependent_update_projection'])) for r in ROLES for k in MODEL])}

逐 probe 的 `I_raw`、归一化交互、mixer 交互范数及 raw→norm 余弦保存在 `local_*.parquet`。Qwen 的大幅度比与 Falcon 的小幅度比表明读后变换的尺度效应显著不同；该比例在 raw 很小时会很大，不能单独证明“非线性创造了大部分全模型纠正”。在两模型均失败的下游恢复门槛下，V36-G 不确认。
"""
    reports["V36_UPSTREAM_CONTEXT_DEPENDENCE.md"] = f"""# V36 上游上下文依赖

NATURAL_FORWARD 让 AA/BA/AB/BB 各有自然上游输入；FIXED_LOCAL_INPUT 将四格都置于 Conv-only 的同一个 mixer 输入。记录两种交互的逐 probe 范数与余弦，并以状态中位为单位。

{_table(('角色','模型','固定/自然交互范数中位比','局部下游收益恢复率'), [(r,MODEL[k],_f(local[r]['models'][k]['fixed_to_natural_interaction_median']),_f(local[r]['models'][k]['whole']['median_local_downstream_fraction'])) for r in ROLES for k in MODEL])}

固定输入交互显著缩小，是上游变化很重要的证据；但未在协议中冻结 V36-C 的单独充分门槛，因此只作描述性支持，不正式宣称上游路径单独解释了全部条件收益。
"""
    reports["V36_NATIVE_OPERATOR_COMPOSITION.md"] = """# V36 原生算子跨层组合

未执行。冻结规则要求局部算子律在两模型开发和验证通过预测、匹配优势、下游插入与家族联合门槛后才开启跨 24 recurrent 层的自然组合。两模型均未达到这些门槛；因此不把历史全深度 read-output replacement 重新包装成原生算子组合，也不复制别的轨迹的下游输出。
"""
    reports["V36_COUNTERFACTUAL_OPERATOR_SWAP.md"] = """# V36 状态×算子反事实交换

已执行固定 h 下 A/B/C 的局部状态×Conv 天然缓存互换，并验证请求/实现哈希及原生重放。`STATE_B+OPERATOR_A` 与 `STATE_A+OPERATOR_B` 不能仅凭张量不同判定未来行为归属。因局部匹配/下游门槛失败，作为正式候选的整模型未来交换与完整九格未来兼容矩阵未开启；不存在可报告的“匹配配对恢复供体未来”结论。
"""
    reports["V36_TASK_GROUNDED_FUNCTIONAL_TEST.md"] = """# V36 任务语义功能测试

未执行。协议只在两模型同一计算机制经开发和验证通过后允许用外部定义的布尔值、绑定、状态转换和模值，测正确答案 logit margin、分类准确率及 donor/recipient 响应忠实度。本版无合格机制候选，独立最终也未开启；不能由五个任务家族的向量响应直接推断语义变量恢复。
"""
    reports["V36_CROSS_MODEL_OPERATOR_LAW.md"] = f"""# V36 跨模型读算子法则

Qwen 与 Falcon 各自的原生公式均对固定局部输入下的旧 REC 状态差预测精确；这是架构内计算等价，不是两模型共享的行为机制确认。

{local_table}

两模型开发/验证均未达到匹配优势和下游插入门槛，且读后归一化尺度方向不同。因此 V36-A 与 H 不成立为已确认结论；不能推到第三模型或应用。
"""
    reports["V36_STRICT_INTERFACE_AUDIT.md"] = f"""# V36 严格接口与对照审计

控制 A：校准完整一 token 原生重放 bitwise；每模型 20 状态、四层、两 REC 状态的公式 raw read 零误差。B/C/I：正式 factorial 的 recipient REC、donor REC × recipient/donor Conv 与自然 joint。D/E：固定 h 的第三自然 token Conv 与 REC 缓存九格，写回哈希在 proof 记录。F：Qwen q/k、Falcon C 的打乱算子组件，仅作为非天然算子诊断。G：合成反号 ΔS 的代数检查，不冒称自然 token。H：自然供体 mixer output 放入 Conv-only 支路作为单层上界，不冒称机制。所有运行均保持 recipient-native KV 与同一六 probe。F/G 不在自然状态流形，不能拿它们的输出当因果任务结果。

{control_table}

请求/实现 REC、Conv、mixer 哈希、local h 哈希、完整自然支路响应哈希和独立最终封存记录分别见 `local_proof_*`、`controls_proof_*`、factorial audit 与 `final_opening_v36.json`。未把 activation similarity 作为机制判据。
"""
    panel = _read(root, "panel_v36.json")
    reports["V36_EXECUTION_MANIFEST.md"] = f"""# V36 执行清单

父提交 `{cfg['parent_commit']}`；基础协议 SHA-256 `{sha256_file(root/'artifacts/computational_origin_v36.freeze.json')}`。语义配对且排除 V28–V35 的样本：校准 `{len(panel['calibration'])}`、开发 `{len(panel['development'])}`、验证 `{len(panel['validation'])}`、封存独立最终 `{len(panel['independent_final'])}`，两模型各同样数量。状态分组哈希：`{json.dumps(panel['role_hashes'], ensure_ascii=False, sort_keys=True)}`。实际 Qwen/Falcon 层位在局部因子报告；token triple、probe、REC/Conv、原生因子与插入请求/实现摘要在机器记录与索引中。

V35 审计结局 `{audit_outcome}`；开发/验证高层重确认均通过；局部候选两角色均未通过；V36 限定范围结局 `{[k for k,v in adjud['formal_outcomes'].items() if v]}`；最终开放记录 `opened=false`，40+40 最终状态从未产生响应。跨层组合、语义任务测试依冻结规则不执行。没有调门槛、没有修改历史 V1–V35 记录。
"""
    reports["V36_SCIENTIFIC_ANSWERS.md"] = f"""# V36 35 个科学问题的审慎答案

1–2. V35 Q2+Q3+Q4 四格均通过，但候选构造只收集单段/两段，遗漏已列入 priority 的三段；需追加判定/报告修订，不能追认独立最终。
3–4. Qwen `S'=GS+k⊗beta(v-kᵀGS), r=qᵀS'`；Falcon `S'=dA⊙S+dBx, r=CᵀS'+Dx`，详见原生方程报告。
5–6. Qwen 的 memory、delta、TRUE_UPDATE 依赖旧 S，q/k/v 依赖 Conv；Falcon dBx 不依赖 S，B/C/x 依赖 Conv，dA 来自 h。
7–10. 同输入局部交互存在但相对自然交互很小，单层下游重插入恢复率远低于 0.20；固定 h 后交互衰减，提示上游上下文重要，但不能宣称全部由上游造成。
11–14. 原生公式对未见开发/验证状态、token、五家族的局部 raw 差方向/排序精确；这是源代码等价，非未来行为预测充分性。
15–17. donor Conv 并非稳定最强匹配；第三自然 token 不支持预定 ≥80% 优势。局部九格曾执行但未保存完整矩阵；未来九格未授权，不能宣称对角兼容。
18–20. 固定输入下 Falcon REC 差由旧状态路径进入，Qwen 旧状态也改变状态依赖移除；范数/投影不能单独确认全局 D/E/F。
21. 归一化尺度效应两模型不同；无跨模型充分下游中介证据，不确认 G。
22–23. Qwen TRUE_UPDATE 依赖旧 S，单独移植漏掉/错配旧状态；Falcon dBx 固定输入与旧 S 独立。依赖一致重算解释为什么单独替换不等于完整条件效应，不把历史失败解读为更新毫无作用。
24–28. 两模型局部代数都成立，但匹配优越/下游充分性失败；不能宣称跨模型读算子行为律或预先预测未来 donor 纠正。
29. 外部定义的任务变量测试未获授权，未测。
30–34. {outcome}
35. 下一模型/应用不能冻结“匹配 read operator”或单层局部机制；可保留已证实的原生局部方程与 V34/V35 全读出切面，下一步需预先设计上游输入/跨层交互实验，且重新冻结新独立样本。
"""
    reports["V36_COMPLETE_REPORT.md"] = f"""# V36 Complete Report — Computational Origin of REC–Conv Conditional Effects

**Does Conv Change How Persistent Recurrent State Is Read?**

结论：原生递推式对两架构固定输入的 REC 状态差读出预测精确，但所测四个预定层既没有稳健的供体 Conv 匹配优势，也无法单层重现大部分全模型条件效应。V36 的强计算机制假说未通过开发/验证联合门槛；独立最终保持封存。不能把局部代数恒等式当成跨模型行为机制。

## V35 审计（与 V36 结局分离）

`{audit_outcome}`：V35 Q2+Q3+Q4 开发/验证两模型强门槛通过，但候选生成遗漏三段组合；已有 FULL_DEPTH_ONLY 最终不检验它。修订只追加，历史文件不覆盖。

## 新面板高层现象

{high_table}

## 局部机制主判定

{local_table}

预定匹配优势 ≥0.80、单层下游恢复 ≥0.20、至少 4/5 家族；两模型两角色均失败。固定/自然交互范数比在开发 Q/F 为 `{_f(local['development']['models']['Q']['fixed_to_natural_interaction_median'])}`/`{_f(local['development']['models']['F']['fixed_to_natural_interaction_median'])}`，验证为 `{_f(local['validation']['models']['Q']['fixed_to_natural_interaction_median'])}`/`{_f(local['validation']['models']['F']['fixed_to_natural_interaction_median'])}`。这支持强上游依赖的描述性判断，但未建立新的全深度计算机制。

## 原生代数、对照和解释边界

两模型各 20 校准状态、160 原始读出对照零误差；开发/验证预测余弦与幅度排序均达到 1.0。Qwen TRUE_UPDATE 旧状态依赖，Falcon dBx 固定输入下旧状态独立；读后归一化在两架构尺度相反。供体 mixer 拷贝为接口上界，不是机制。全部对照与 hash 在专题报告/机器索引。

## 封存与后继

最终开放记录 `opened=false`，40+40 独立最终响应未观察。跨层自然组合、未来 3×3、外部语义任务按协议不授权。不能确认 V36-A/B/D/E/F/G/H；C 为描述性上游依赖，I 只限“已测试局部候选不足”，不得泛化为不存在其他计算律。下一阶段需冻结新的跨层/上游上下文假说与新样本，保留 V34/V35 全读出因果切面，不称显式误差纠正。
"""
    for name, body in reports.items():
        _write_once(root, name, body)
    final_path = root / REPORTS / "FINAL_REPORT.md"
    final_text = final_path.read_text(encoding="utf-8")
    if "<!-- V36_START -->" not in final_text:
        section = """\n\n<!-- V36_START -->\n## V36 — Computational Origin of REC–Conv Conditional Effects\n\nV35 审计追加修订：Q2+Q3+Q4 在开发/验证两模型均过强门槛，但候选生成遗漏，历史最终不能追认。V36 新面板高层 REC×Conv 条件收益再次通过；两模型原生固定输入公式对局部 raw read 零误差。然而供体 Conv 匹配优势与四层单层下游恢复率都低于预定门槛，未找到可进入独立最终的共同机制。40+40 最终状态保持封存；跨层组合与语义任务测试未授权。详见 `reports/V36_COMPLETE_REPORT.md` 和 `reports/V36_ALL_REPORTS.md`。\n<!-- V36_END -->\n"""
        final_path.write_text(final_text.rstrip() + section, encoding="utf-8")
    elif "<!-- V36_END -->" not in final_text:
        raise RuntimeError("V36 FINAL_REPORT section incomplete")
    report_paths = sorted((root / REPORTS).glob("V36_*.md"))
    bundle_lines = ["# V36 All Reports — Computational Origin of REC–Conv Conditional Effects", "",
                    "This is the standalone single-file bundle for V36; topic reports remain separately preserved.", "",
                    "## Topic report hashes", "",
                    _table(("Report", "SHA-256"), [(p.name, sha256_file(p)) for p in report_paths if p.name != "V36_ALL_REPORTS.md"])]
    for path in report_paths:
        if path.name != "V36_ALL_REPORTS.md":
            bundle_lines.extend(["", "---", "", f"## {path.name}", "", path.read_text(encoding="utf-8").rstrip()])
    all_path = _write_once(root, "V36_ALL_REPORTS.md", "\n".join(bundle_lines))
    indexed_results = sorted(p for p in (root / OUT).iterdir() if p.is_file() and p.name != "v36_integrity_index.json")
    index = {"version": "V36", "base_freeze_sha256": sha256_file(root / "artifacts/computational_origin_v36.freeze.json"),
             "final_opening_sha256": sha256_file(root / OUT / "final_opening_v36.json"),
             "final_opened": False,
             "results": {str(p.relative_to(root)): sha256_file(p) for p in indexed_results},
             "reports": {str(p.relative_to(root)): sha256_file(p) for p in sorted((root/REPORTS).glob("V36_*.md"))},
             "cumulative_report_snapshot_sha256": sha256_file(final_path),
             "historical_files_modified": False}
    index_path = root / OUT / "v36_integrity_index.json"
    write_json_atomic(index_path, index)
    seal = stage_freeze(root, "report", [SOURCE, str(index_path.relative_to(root)),
                        *(str(p.relative_to(root)) for p in sorted((root/REPORTS).glob("V36_*.md")))],
                        {"report_count": len(index["reports"]), "result_count": len(index["results"]),
                         "index_sha256": sha256_file(index_path), "final_opened": False})
    return {"freeze_digest": seal["freeze_digest"], "report_count": len(index["reports"]),
            "result_count": len(index["results"]), "all_reports": str(all_path.relative_to(root))}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), ensure_ascii=False, indent=2))
