# Support-guided Hyperloop files for HyperSynergy

把这三个文件复制到 `~/HyperSynergy/`：

- `model_loop.py`
- `train_loop.py`
- `test_loop.py`

## 训练

建议先小规模跑通：

```bash
cd ~/HyperSynergy
conda activate m3net
mkdir -p logs saved_model/saved_model_loop_setting loop_outputs

nohup env \
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
PYTHONFAULTHANDLER=1 \
CUDA_LAUNCH_BLOCKING=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
python -u train_loop.py --epochs 1000 --support 50 --query 40 --validate_every 50 --print_every 10 \
> logs/train_loop.log 2>&1 &

tail -n 200 -f logs/train_loop.log
```

训练时终端会直接输出：

- train loss / MSE / RMSE / SCC / PCC / R²
- val loss / MSE / RMSE / SCC / PCC / R²
- best_epoch
- best_val_mse
- checkpoint 路径

同时保存 CSV：

```bash
loop_outputs/train_loop_metrics.csv
```

## 测试

```bash
nohup env \
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
PYTHONFAULTHANDLER=1 \
CUDA_LAUNCH_BLOCKING=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
python -u test_loop.py --support 50 --query 40 --test_batches 200 --repeats 5 \
> logs/test_loop.log 2>&1 &

tail -n 200 -f logs/test_loop.log
```

测试会自动读取 `saved_model/saved_model_loop_setting/` 里 MSE 最小的 checkpoint。

测试输出会直接显示：

- 每个 repeat 的 MSE / RMSE / SCC / PCC / R²
- 最终 mean 和 std

保存位置：

```bash
loop_outputs/test_results/test_summary.json
loop_outputs/test_results/test_repeat_metrics.csv
loop_outputs/test_results/preds_first_repeat.csv
loop_outputs/test_results/labels_first_repeat.csv
```

## 备注

这个版本保留 HyperSynergy 的原始 `Feature_embedding_network`，可以加载你已经训练好的：

```bash
pretrain_representation_model/pretrain_representation_few_zero_setting.model
```

核心替换为：

```text
Support prototype + loop-wise Transformer refinement + support-controlled gate
```
