import os
import time
import json
import pickle
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from utils_syn import *
import data_syn
from model_hypersynergy import HyperSynergy
from research_utils import calc_metrics, safe_corr, set_seed

warnings.filterwarnings("ignore")
torch.set_num_threads(2)




def make_task_batch(mini, mode, drug_features, cell_features, codes, device):
    if mode == "train":
        x_support_set, x_target = mini.get_train_batch(augment=False)
    else:
        x_support_set, x_target = mini.get_test_batch(augment=False)

    b, spc, t = np.shape(x_support_set)
    bt, spct, tt = np.shape(x_target)

    support_set_samples = x_support_set.reshape(b, spc, t)
    target_samples = x_target.reshape(bt, spct, tt)

    support_target_samples = np.concatenate([support_set_samples, target_samples], axis=1)
    b, s, t = np.shape(support_target_samples)
    support_target_samples = support_target_samples.reshape(b * s, t)
    num_samples = len(support_target_samples)

    train_data = TestbedDataset(support_target_samples, drug_features, cell_features, codes)
    train_data1 = TestbedDataset1(support_target_samples, drug_features, cell_features, codes)

    loader = DataLoader(train_data, batch_size=num_samples, shuffle=False, num_workers=0)
    loader1 = DataLoader(train_data1, batch_size=num_samples, shuffle=False, num_workers=0)

    data = next(iter(loader)).to(device)
    data1 = next(iter(loader1)).to(device)
    return data, data1, b, num_samples


@torch.no_grad()
def evaluate(model, mini, drug_features, cell_features, codes, device, val_batches):
    model.eval()
    losses, all_preds, all_labels = [], [], []

    for _ in range(val_batches):
        data, data1, batch_size, _ = make_task_batch(mini, "val", drug_features, cell_features, codes, device)
        val_loss, _, _, query_loss, output, target = model.run_batch(data, data1, batch_size, False)
        losses.append(float(query_loss.detach().cpu().item()))
        all_preds.append(output.detach().cpu().numpy().reshape(-1))
        all_labels.append(target.detach().cpu().numpy().reshape(-1))

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    metrics = calc_metrics(labels, preds)
    metrics["loss"] = float(np.mean(losses))
    return metrics, preds, labels


def load_pretrained_extractor(model, model_path, device):
    pretrained_dict = torch.load(model_path, map_location=device)
    model_dict = model.extractor.state_dict()
    state_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(state_dict)
    model.extractor.load_state_dict(model_dict)
    for p in model.extractor.parameters():
        p.requires_grad = False
    return len(state_dict)


def append_csv(path, row, header):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8") as f:
        if not exists:
            f.write(",".join(header) + "\n")
        f.write(",".join(str(row.get(h, "")) for h in header) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--support", type=int, default=50)
    parser.add_argument("--query", type=int, default=40)
    parser.add_argument("--val_batches", type=int, default=30)
    parser.add_argument("--validate_every", type=int, default=50)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loops", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--cuda", type=str, default="cuda:0")
    parser.add_argument("--feature_path", type=str, default="data/sample_features/")
    parser.add_argument("--pretrained", type=str, default="./pretrain_representation_model/pretrain_representation_few_zero_setting.model")
    parser.add_argument("--save_dir", type=str, default="saved_model/saved_model_loop_setting/")
    parser.add_argument("--result_csv", type=str, default="loop_outputs/train_loop_metrics.csv")
    args = parser.parse_args()
    set_seed(args.seed)

    device = args.cuda if torch.cuda.is_available() else "cpu"
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path("loop_outputs").mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Support-guided Hyperloop few-shot training")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))
    print("device:", device)
    print("=" * 80, flush=True)

    with open(os.path.join(args.feature_path, "codes_cell.p"), "rb") as f:
        codes = pickle.load(f)
    with open(os.path.join(args.feature_path, "drug_feature_cell.p"), "rb") as f:
        drug_features = pickle.load(f)
    with open(os.path.join(args.feature_path, "cell_feature_900.p"), "rb") as f:
        cell_features = pickle.load(f)

    mini = data_syn.MiniCellDataSet(batch_size=args.batch_size, samples_support=args.support, samples_query=args.query)

    model = HyperSynergy(num_support=args.support, num_query=args.query, n_loops=args.loops, n_heads=args.heads, dropout=args.dropout).to(device)
    loaded = load_pretrained_extractor(model, args.pretrained, device)
    print(f"Loaded pretrained extractor tensors: {loaded}", flush=True)

    lr_list = ["inner_l_rate"]
    params = [x[1] for x in model.named_parameters() if x[0] not in lr_list and x[1].requires_grad]
    lr_params = [x[1] for x in model.named_parameters() if x[0] in lr_list and x[1].requires_grad]

    optimizer = optim.Adam(params, lr=args.lr, weight_decay=1.0e-5)
    optimizer_lr = optim.Adam(lr_params, lr=0.1) if lr_params else None
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 2500, 5000, 10000, 30000], gamma=0.5)

    header = ["time", "epoch", "split", "lr", "loss", "mse", "rmse", "scc", "pcc", "r2", "best_epoch", "best_val_mse", "checkpoint"]

    best_val_mse = float("inf")
    best_epoch = -1
    best_path = ""
    train_losses, train_preds, train_labels = [], [], []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        if optimizer_lr:
            optimizer_lr.zero_grad()

        data, data1, task_batch, _ = make_task_batch(mini, "train", drug_features, cell_features, codes, device)
        total_loss, _, _, query_loss, query_output, query_target = model.run_batch(data, data1, task_batch, True)

        total_loss.backward()
        optimizer.step()
        if optimizer_lr:
            optimizer_lr.step()
        if scheduler.get_last_lr()[0] > 1e-7:
            scheduler.step()

        train_losses.append(float(query_loss.detach().cpu().item()))
        train_preds.append(query_output.detach().cpu().numpy().reshape(-1))
        train_labels.append(query_target.detach().cpu().numpy().reshape(-1))

        if epoch % args.print_every == 0:
            labels = np.concatenate(train_labels)
            preds = np.concatenate(train_preds)
            m = calc_metrics(labels, preds)
            avg_loss = float(np.mean(train_losses))
            elapsed = time.time() - start
            print(
                f"[TRAIN] epoch={epoch:05d}/{args.epochs} loss={avg_loss:.6f} mse={m['mse']:.6f} "
                f"rmse={m['rmse']:.6f} scc={m['scc']:.4f} pcc={m['pcc']:.4f} r2={m['r2']:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} elapsed={elapsed/60:.1f}min",
                flush=True,
            )
            append_csv(args.result_csv, {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "epoch": epoch, "split": "train", "lr": scheduler.get_last_lr()[0], "loss": avg_loss, **m, "best_epoch": best_epoch, "best_val_mse": best_val_mse, "checkpoint": best_path}, header)
            train_losses.clear(); train_preds.clear(); train_labels.clear()

        if epoch % args.validate_every == 0:
            val_metrics, _, _ = evaluate(model, mini, drug_features, cell_features, codes, device, args.val_batches)
            improved = val_metrics["mse"] < best_val_mse
            if improved:
                best_val_mse = val_metrics["mse"]
                best_epoch = epoch
                best_path = os.path.join(args.save_dir, f"loop_epoch{epoch}_mse{best_val_mse:.6f}.pth")
                torch.save({"epoch": epoch, "state_dict": model.state_dict(), "args": vars(args), "val_metrics": val_metrics}, best_path)

            print(
                f"[VAL]   epoch={epoch:05d}/{args.epochs} loss={val_metrics['loss']:.6f} mse={val_metrics['mse']:.6f} "
                f"rmse={val_metrics['rmse']:.6f} scc={val_metrics['scc']:.4f} pcc={val_metrics['pcc']:.4f} r2={val_metrics['r2']:.4f} "
                f"best_epoch={best_epoch} best_mse={best_val_mse:.6f} saved={best_path if improved else 'no'}",
                flush=True,
            )
            append_csv(args.result_csv, {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "epoch": epoch, "split": "val", "lr": scheduler.get_last_lr()[0], "loss": val_metrics["loss"], **val_metrics, "best_epoch": best_epoch, "best_val_mse": best_val_mse, "checkpoint": best_path}, header)

    print("=" * 80)
    print(f"Training finished. best_epoch={best_epoch}, best_val_mse={best_val_mse:.6f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Metrics CSV: {args.result_csv}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
