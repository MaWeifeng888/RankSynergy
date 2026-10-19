# RankSynergy

Few-shot drug synergy prediction with support-guided adaptation and ranking residuals.

## 当前状态

这是服务器实验代码的整理副本，不是完整的独立运行环境。原始 E 盘目录未修改。
本仓库不声明新增算法创新或性能提升；已有图中的数值来自原绘图脚本，尚未通过实验日志核验。

## 内容

- `model/ranksynergy.py`：交叉注意力、排序残差及稀疏适配器的实验实现。
- `model/model_hypersynergy.py`：支持集原型、共享 Transformer 迭代细化和门控融合。
- `model/train_hypersynergy.py`、`model/test_hypersynergy.py`：训练与重复评估入口。
- `data/`：当前提供的特征和数据划分；公开前须确认数据来源及再分发许可。
- `paper_figs/`、`tu/`：已有图与压缩归档，保留用于对应实验材料。
- `plot_ranksynergy_figures.py`：包含硬编码汇总值的绘图脚本，不用于生成实验结果。

## 服务器依赖

以下文件不在当前副本中，需要从实际运行服务器找回：

- `data_syn.py`
- `utils_syn.py`
- `model_few.py`
- `train_v13_transrank_adapter.py`
- `train_loop_v40_hard_rank_adapter.py`
- 对应预训练特征提取器及评估 checkpoint。

`requirements.txt` 只列当前可见代码的主要第三方依赖，不是经过验证的锁定环境。
请从服务器导出真实环境版本，并确认上述依赖的许可证和引用要求。

补齐依赖后，在仓库根目录执行：

```bash
python model/train_hypersynergy.py --seed 2026
python model/test_hypersynergy.py --seed 2026 --checkpoint PATH_TO_CHECKPOINT
```

`model/README_loop.md` 是服务器旧版使用笔记，其旧文件名不能直接用于当前目录。

## 本次整理与科研复现注意事项

训练与测试导入已对齐实际模型文件，新增共享指标实现及 `--seed` 参数。
Spearman 相关采用平均秩处理并列值；无效输入明确报错。为兼容原代码，常量输入的相关系数及 R² 约定返回 0。
固定种子能改善可复现性，但不能保证跨硬件或所有算子的逐位一致。

当前训练脚本的验证过程调用 `get_test_batch`。需要结合服务器的 `data_syn.py` 确认其真实划分，
确保 checkpoint 选择使用独立验证集，最终测试集只用于最终报告。
RankSynergy 内部选择 checkpoint 同样依赖后端的划分实现，必须一并核验。
不要在未核验划分前把结果作为独立测试集性能。

建议后续实验分别去除排序损失、相关损失、支持门控及稀疏选择，并保持相同数据划分与随机种子。
保存每个种子和每个细胞系的原始预测、配置、环境和 checkpoint，再进行配对统计分析。
这些是待完成的验证计划，不代表已经获得实验结论。

当前未添加开源许可证：需先核实原始 HyperSynergy 代码和数据的授权来源。
