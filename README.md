# RankSynergy

Few-shot drug synergy prediction with support-guided adaptation and ranking residuals.


## 内容

- `model/ranksynergy.py`：交叉注意力、排序残差及稀疏适配器的实验实现。
- `model/model_hypersynergy.py`：支持集原型、共享 Transformer 迭代细化和门控融合。
- `model/train_hypersynergy.py`、`model/test_hypersynergy.py`：训练与重复评估入口。
- `data/`：当前提供的特征和数据划分；公开前须确认数据来源及再分发许可。
- `paper_figs/`、`tu/`：已有图与压缩归档，保留用于对应实验材料。
- `plot_ranksynergy_figures.py`：包含硬编码汇总值的绘图脚本，不用于生成实验结果。



