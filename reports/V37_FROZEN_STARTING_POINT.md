# V37 冻结起点

V37 父提交 `d7b93f041706a60895f0294c9e75c9cb6363a4f6` 为已推送 V36；V1–V36 正式文件和结果完全不变，V36 正式观测不计入 V37。仅复用经新校准审计的工程 instrumentation/方程。

新样本池：校准 `20`、开发 `80`、正式验证 `40`、独立最终 `40` states/model。验证和最终各有独立预封存池；V18 training-source 状态不进入 V37 formal validation/final。五任务族各为 4/16/8/8 states。池封存时两模型响应均未观测。样本为同任务模板的新程序/提示，不宣称任务族分布外泛化。

样本分组哈希：`{"calibration": "607055d330875b4a2be11f5e18e3a05163214cc25587797ff266d79e9e7ad117", "development": "624ed4fc15b5db01560489de777eba891080c6e178b500efcfb78b86f0262381", "independent_final": "f3d4a7839755f2321dbab49ab00726713062a2f9f4d4eaca3092d634122615b3", "validation": "687a97b08ef1619cfc51a5cafdccf10b454b9ef8ee3d3341378944a89381029a"}`。源审计：`fa8dfc684d9ff5772f6c7b6248884ae1d016e0da378c6694e50c6124a0433be8`。
