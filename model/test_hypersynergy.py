import os
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils_syn import *
import data_syn
from model_hypersynergy import HyperSynergy
from research_utils import calc_metrics, safe_corr, set_seed




def make_task_batch(mini, drug_features, cell_features, codes, device):
    x_support_set, x_target = mini.get_test_batch(augment=False)

    b, spc, t = np.shape(x_support_set)
    bt, spct, tt = np.shape(x_target)

    support_set_samples = x_support_set.reshape(b, spc, t)
    target_samples = x_target.reshape(bt, spct, tt)

    support_target_samples = np.concatenate([support_set_samples, target_samples], axis=1)
    b, s, t = np.shape(support_target_samples)

    support_target_samples = support_target_samples.reshape(b * s, t)
    num_samples = len(support_target_samples)

    test_data = TestbedDataset(support_target_samples, drug_features, cell_features, codes)
    test_data1 = TestbedDataset1(support_target_samples, drug_features, cell_features, codes)

    loader = DataLoader(test_data, batch_size=num_samples, shuffle=False, num_workers=0)
    loader1 = DataLoader(test_data1, batch_size=num_samples, shuffle=False, num_workers=0)

    data = next(iter(loader)).to(device)
    data1 = next(iter(loader1)).to(device)
    return data, data1, b


def find_best_checkpoint(save_dir):
    save_dir = Path(save_dir)
    ckpts = sorted(save_dir.glob("loop_epoch*_mse*.pth"))
    if not ckpts:
        ckpts = sorted(save_dir.glob("*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {save_dir}")

    def mse_from_name(p):
        name = p.name
        if "_mse" in name:
            try:
                return float(name.split("_mse")[-1].replace(".pth", ""))
            except Exception:
                pass
        return float("inf")

    return str(sorted(ckpts, key=mse_from_name)[0])


@torch.no_grad()
def test_repeats(model, mini, drug_features, cell_features, codes, device, test_batches, repeats):
    model.eval()
    repeat_metrics = []
    all_preds_for_save = None
    all_labels_for_save = None

    for r in range(1, repeats + 1):
        losses, all_preds, all_labels = [], [], []

        for step in range(1, test_batches + 1):
            data, data1, task_batch = make_task_batch(mini, drug_features, cell_features, codes, device)
            _, _, _, test_loss, output, target = model.run_batch(data, data1, task_batch, False)

            losses.append(float(test_loss.detach().cpu().item()))
            all_preds.append(output.detach().cpu().numpy().reshape(-1))
            all_labels.append(target.detach().cpu().numpy().reshape(-1))

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        metrics = calc_metrics(labels, preds)
        metrics["loss"] = float(np.mean(losses))
        repeat_metrics.append(metrics)

        if all_preds_for_save is None:
            all_preds_for_save = preds
            all_labels_for_save = labels

        print(
            f"[TEST repeat {r:02d}/{repeats}] loss={metrics['loss']:.6f} mse={metrics['mse']:.6f} "
            f"rmse={metrics['rmse']:.6f} scc={metrics['scc']:.4f} pcc={metrics['pcc']:.4f} r2={metrics['r2']:.4f}",
            flush=True,
        )

    summary = {}
    for k in ["loss", "mse", "rmse", "scc", "pcc", "r2"]:
        vals = np.array([m[k] for m in repeat_metrics], dtype=float)
        summary[k + "_mean"] = float(vals.mean())
        summary[k + "_std"] = float(vals.std())

    return summary, repeat_metrics, all_preds_for_save, all_labels_for_save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--support", type=int, default=50)
    parser.add_argument("--query", type=int, default=40)
    parser.add_argument("--test_batches", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--loops", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--cuda", type=str, default="cuda:0")
    parser.add_argument("--feature_path", type=str, default="data/sample_features/")
    parser.add_argument("--save_dir", type=str, default="saved_model/saved_model_loop_setting/")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="loop_outputs/test_results")
    args = parser.parse_args()
    set_seed(args.seed)

    device = args.cuda if torch.cuda.is_available() else "cpu"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ckpt = args.checkpoint.strip() or find_best_checkpoint(args.save_dir)

    print("=" * 80)
    print("Support-guided Hyperloop few-shot testing")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))
    print("device:", device)
    print("checkpoint:", ckpt)
    print("=" * 80, flush=True)

    with open(os.path.join(args.feature_path, "codes_cell.p"), "rb") as f:
        codes = pickle.load(f)
    with open(os.path.join(args.feature_path, "drug_feature_cell.p"), "rb") as f:
        drug_features = pickle.load(f)
    with open(os.path.join(args.feature_path, "cell_feature_900.p"), "rb") as f:
        cell_features = pickle.load(f)

    mini = data_syn.MiniCellDataSet(batch_size=args.batch_size, samples_support=args.support, samples_query=args.query)
    model = HyperSynergy(num_support=args.support, num_query=args.query, n_loops=args.loops, n_heads=args.heads, dropout=args.dropout).to(device)

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["state_dict"])

    summary, repeat_metrics, preds, labels = test_repeats(model, mini, drug_features, cell_features, codes, device, args.test_batches, args.repeats)

    print("=" * 80)
    print("[FINAL TEST SUMMARY]")
    for k, v in summary.items():
        print(f"{k}: {v:.6f}")
    print("=" * 80, flush=True)

    out_dir = Path(args.out_dir)
    np.savetxt(out_dir / "preds_first_repeat.csv", preds, delimiter=",")
    np.savetxt(out_dir / "labels_first_repeat.csv", labels, delimiter=",")

    with (out_dir / "test_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"checkpoint": ckpt, "args": vars(args), "summary": summary, "repeats": repeat_metrics}, f, indent=2, ensure_ascii=False)

    csv_path = out_dir / "test_repeat_metrics.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("repeat,loss,mse,rmse,scc,pcc,r2\n")
        for i, m in enumerate(repeat_metrics, start=1):
            f.write(f"{i},{m['loss']},{m['mse']},{m['rmse']},{m['scc']},{m['pcc']},{m['r2']}\n")

    print(f"Saved summary to: {out_dir / 'test_summary.json'}")
    print(f"Saved repeat metrics to: {csv_path}")


if __name__ == "__main__":
    main()
