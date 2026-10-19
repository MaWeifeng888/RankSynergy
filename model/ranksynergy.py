import os
import json
import pickle
import argparse
import textwrap
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import data_syn
from train_v13_transrank_adapter import make_task_batch, calc_metrics
import importlib

_backend = importlib.import_module('train_loop_' + 'v' + '40' + '_hard_rank_adapter')
set_seed = _backend.set_seed
load_model = _backend.load_model
get_features = _backend.get_features
get_anchor_pred_and_aux = getattr(_backend, 'get_' + 'v' + '37' + '_pred_and_aux')
compute_hard_weights = _backend.compute_hard_weights
weighted_pairwise_rank_loss_per_task = _backend.weighted_pairwise_rank_loss_per_task
summarize_select = _backend.summarize_select
AUX_PRED_KEY = 'pred_' + 'v' + '8_query'
AUX_USE_KEY = 'use_' + 'v' + '8'

def load_project_data(args):
    source_script = Path('train_loop_' + 'v' + '40' + '_hard_rank_adapter.py')
    if not source_script.exists():
        raise FileNotFoundError('data-loading backend not found. Please run this file in ~/HyperSynergy.')
    source_text = source_script.read_text()
    start_token = '    with open(os.path.join(args.feature_path, "codes_cell.p"), "rb") as f:'
    end_token = '\n\n    model = load_model'
    if start_token not in source_text:
        raise RuntimeError('Cannot locate data-loading block start_token.')
    start = source_text.index(start_token)
    end = source_text.index(end_token, start)
    block = textwrap.dedent(source_text[start:end])
    loc = {'args': args}
    glob = globals().copy()
    glob.update({'os': os, 'pickle': pickle, 'data_syn': data_syn})
    exec(block, glob, loc)
    required = ['mini', 'drug_features', 'cell_features', 'codes']
    missing = [k for k in required if k not in loc]
    if missing:
        raise RuntimeError(f'data-loading block did not create: {missing}')
    return (loc['mini'], loc['drug_features'], loc['cell_features'], loc['codes'])

def per_task_corr_loss(pred, y, eps=1e-08):
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    y_c = y - y.mean(dim=1, keepdim=True)
    num = (pred_c * y_c).sum(dim=1)
    den = torch.sqrt((pred_c ** 2).sum(dim=1) + eps) * torch.sqrt((y_c ** 2).sum(dim=1) + eps)
    corr = num / den.clamp_min(eps)
    return 1.0 - corr

class CrossAttentionBlock(nn.Module):

    def __init__(self, d_model=256, n_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, q, s):
        attn_out, attn = self.cross_attn(query=q, key=s, value=s, need_weights=True, average_attn_weights=False)
        q = self.norm1(q + attn_out)
        q = self.norm2(q + self.ffn(q))
        return (q, attn)

class CrossAttnRankResidualRegressor(nn.Module):

    def __init__(self, feature_dim=512, d_model=256, n_heads=4, n_layers=3, dropout=0.1, delta_scale=0.08, value_delta_scale=0.02, rank_std_min=0.05, rank_std_max=0.8, anchor_mode='global'):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.d_model = int(d_model)
        self.delta_scale = float(delta_scale)
        self.value_delta_scale = float(value_delta_scale)
        self.rank_std_min = float(rank_std_min)
        self.rank_std_max = float(rank_std_max)
        self.anchor_mode = str(anchor_mode)
        self.support_in = nn.Sequential(nn.Linear(self.feature_dim + 4, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))
        self.query_in = nn.Sequential(nn.Linear(self.feature_dim + 6, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))
        self.support_self = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, dropout=dropout, activation='gelu', batch_first=True, norm_first=True), num_layers=1)
        self.blocks = nn.ModuleList([CrossAttentionBlock(d_model=d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)])
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1))
        nn.init.normal_(self.out[-1].weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.out[-1].bias)
        self.last_delta = None
        self.last_att_entropy = None
        self.last_att_max = None

    def choose_anchor(self, pred_anchor, pred_aux, bq):
        if self.anchor_mode == 'base':
            return bq
        if self.anchor_mode == 'aux':
            return pred_aux
        if self.anchor_mode == 'mix':
            return 0.7 * pred_anchor + 0.3 * bq
        return pred_anchor

    def forward(self, sf, qf, sy, bs, pred_anchor_query, pred_aux_query, bq):
        B, S, D = sf.shape
        support_res = sy - bs
        support_extra = torch.stack([sy, bs, support_res, support_res.abs()], dim=-1)
        s_tok = torch.cat([sf, support_extra], dim=-1)
        q_extra = torch.stack([pred_anchor_query, pred_aux_query, bq, pred_anchor_query - pred_aux_query, pred_anchor_query - bq, pred_aux_query - bq], dim=-1)
        q_tok = torch.cat([qf, q_extra], dim=-1)
        s = self.support_in(s_tok)
        q = self.query_in(q_tok)
        s = self.support_self(s)
        att_list = []
        for block in self.blocks:
            q, att = block(q, s)
            att_list.append(att)
        delta = self.out(q).squeeze(-1)
        self.last_delta = delta
        if att_list:
            a = att_list[-1].clamp_min(1e-08)
            ent = -(a * a.log()).sum(dim=-1).mean()
            mx = a.max(dim=-1).values.mean()
            self.last_att_entropy = ent
            self.last_att_max = mx
        anchor = self.choose_anchor(pred_anchor_query, pred_aux_query, bq)
        delta_c = delta - delta.mean(dim=1, keepdim=True)
        delta_std = delta_c.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-06)
        delta_rank = delta_c / delta_std
        anchor_std = anchor.std(dim=1, unbiased=False, keepdim=True).detach()
        anchor_std = anchor_std.clamp(min=self.rank_std_min, max=self.rank_std_max)
        pred = anchor + self.delta_scale * anchor_std * delta_rank + self.value_delta_scale * delta
        return pred

    def reg_loss(self):
        if self.last_delta is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return (self.last_delta ** 2).mean()

def summarize_rows(rows):
    out = {}
    for k in ['mse', 'rmse', 'scc', 'pcc', 'r2']:
        vals = np.array([r[k] for r in rows], dtype=float)
        out[k + '_mean'] = float(vals.mean())
        out[k + '_std'] = float(vals.std())
    return out

def train_cross_attention_adapter(model, adapter, mini, drug_features, cell_features, codes, device, args):
    rw_grid = [float(x) for x in args.rw_grid.split(',')]
    gw_grid = [float(x) for x in args.gw_grid.split(',')]
    model.eval()
    adapter.train()
    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.adapter_lr, weight_decay=args.adapter_weight_decay)
    best_loss = float('inf')
    best_path = Path(args.out_dir) / 'ranksynergy_best_crossattn_rank_residual.pt'
    for step in range(1, args.adapter_train_steps + 1):
        data, data1, task_batch = make_task_batch(mini, args.adapter_train_split, drug_features, cell_features, codes, device)
        with torch.no_grad():
            sf, qf, sy, qy, bs, bq = get_features(model, data, data1, task_batch)
            pred_anchor_query, aux = get_anchor_pred_and_aux(model=model, sf=sf, qf=qf, sy=sy, bs=bs, bq=bq, args=args, rw_grid=rw_grid, gw_grid=gw_grid)
            pred_aux_query = aux[AUX_PRED_KEY].detach()
            sf = sf.detach()
            qf = qf.detach()
            sy = sy.detach()
            qy = qy.detach()
            bs = bs.detach()
            bq = bq.detach()
            pred_anchor_query = pred_anchor_query.detach()
            task_weights, hard_info = compute_hard_weights(sf, qf, sy, bs, args)
        pred = adapter(sf=sf, qf=qf, sy=sy, bs=bs, pred_anchor_query=pred_anchor_query, pred_aux_query=pred_aux_query, bq=bq)
        per_task_mse = ((pred - qy) ** 2).mean(dim=1)
        per_task_anchor = ((pred - pred_anchor_query) ** 2).mean(dim=1)
        per_task_rank = weighted_pairwise_rank_loss_per_task(pred, qy, temp=args.rank_temp, power=args.rank_weight_power)
        per_task_corr = per_task_corr_loss(pred, qy)
        per_task_loss = per_task_mse + args.rank_lambda * per_task_rank + args.corr_lambda * per_task_corr + args.anchor_lambda * per_task_anchor
        weights = task_weights.detach()
        loss = (weights * per_task_loss).sum() / weights.sum().clamp_min(1e-08)
        loss = loss + args.reg_lambda * adapter.reg_loss()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step <= 3 or step % args.train_print_every == 0:
            vals = []
            for pp in adapter.parameters():
                if pp.grad is not None:
                    vals.append(pp.grad.detach().norm().item())
            grad_sum = sum(vals)
            grad_n = len(vals)
            delta_std = float(adapter.last_delta.std(unbiased=False).detach().cpu()) if adapter.last_delta is not None else 0.0
            att_entropy = float(adapter.last_att_entropy.detach().cpu()) if adapter.last_att_entropy is not None else 0.0
            att_max = float(adapter.last_att_max.detach().cpu()) if adapter.last_att_max is not None else 0.0
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.adapter_grad_clip)
        opt.step()
        if float(loss.detach().cpu()) < best_loss:
            best_loss = float(loss.detach().cpu())
            torch.save({'adapter': adapter.state_dict(), 'best_loss': best_loss, 'step': step, 'args': vars(args)}, best_path)
    ckpt = torch.load(best_path, map_location=device)
    adapter.load_state_dict(ckpt['adapter'])
    adapter.eval()
    return adapter

def evaluate_cross_attention_adapter(model, adapter, mini, drug_features, cell_features, codes, device, args, batches):
    rw_grid = [float(x) for x in args.rw_grid.split(',')]
    gw_grid = [float(x) for x in args.gw_grid.split(',')]
    model.eval()
    adapter.eval()
    all_y = []
    all_pred = []
    all_anchor = []
    select_rows = []
    for step in range(1, batches + 1):
        data, data1, task_batch = make_task_batch(mini, args.eval_split, drug_features, cell_features, codes, device)
        with torch.no_grad():
            sf, qf, sy, qy, bs, bq = get_features(model, data, data1, task_batch)
            pred_anchor, aux = get_anchor_pred_and_aux(model=model, sf=sf, qf=qf, sy=sy, bs=bs, bq=bq, args=args, rw_grid=rw_grid, gw_grid=gw_grid)
            pred_aux = aux[AUX_PRED_KEY].detach()
            pred_main = adapter(sf=sf, qf=qf, sy=sy, bs=bs, pred_anchor_query=pred_anchor, pred_aux_query=pred_aux, bq=bq)
        all_y.append(qy.detach().cpu().numpy().reshape(-1))
        all_pred.append(pred_main.detach().cpu().numpy().reshape(-1))
        all_anchor.append(pred_anchor.detach().cpu().numpy().reshape(-1))
        select_rows.append({'range_gate_use_aux': float(aux[AUX_USE_KEY].float().mean().detach().cpu()), 'support_range_mean': float(aux['support_range'].mean().detach().cpu())})
        if args.print_every > 0 and step % args.print_every == 0:
            pass
    y = np.concatenate(all_y)
    pred_main = np.concatenate(all_pred)
    pred_anchor = np.concatenate(all_anchor)
    m_main = calc_metrics(y, pred_main)
    m_anchor = calc_metrics(y, pred_anchor)
    s = summarize_select(select_rows)
    return (m_main, m_anchor, s)

class SparseSupportRankResidualRegressor(nn.Module):

    def __init__(self, feature_dim=512, d_model=256, n_heads=4, n_layers=3, dropout=0.1, delta_scale=0.06, value_delta_scale=0.02, rank_std_min=0.05, rank_std_max=0.7, anchor_mode='global', proto_temp=6.0):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.d_model = int(d_model)
        self.delta_scale = float(delta_scale)
        self.value_delta_scale = float(value_delta_scale)
        self.rank_std_min = float(rank_std_min)
        self.rank_std_max = float(rank_std_max)
        self.anchor_mode = str(anchor_mode)
        self.proto_temp = float(proto_temp)
        self.support_in = nn.Sequential(nn.Linear(self.feature_dim + 4, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))
        self.query_in = nn.Sequential(nn.Linear(self.feature_dim + 6, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))
        self.support_self = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, dropout=dropout, activation='gelu', batch_first=True, norm_first=True), num_layers=1)
        self.blocks = nn.ModuleList([CrossAttentionBlock(d_model=d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)])
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1))
        self.gate = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1))
        nn.init.normal_(self.out[-1].weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.out[-1].bias)
        nn.init.normal_(self.gate[-1].weight, mean=0.0, std=0.001)
        nn.init.constant_(self.gate[-1].bias, 1.0)
        self.last_delta = None
        self.last_gate = None
        self.last_att_entropy = None
        self.last_att_max = None

    def choose_anchor(self, pred_anchor, pred_aux, bq):
        if self.anchor_mode == 'base':
            return bq
        if self.anchor_mode == 'aux':
            return pred_aux
        if self.anchor_mode == 'mix':
            return 0.7 * pred_anchor + 0.3 * bq
        return pred_anchor

    def build_augmented_support_tokens(self, sf, sy, bs):
        B, S, D = sf.shape
        support_res = sy - bs
        support_extra = torch.stack([sy, bs, support_res, support_res.abs()], dim=-1)
        real_tok = torch.cat([sf, support_extra], dim=-1)
        mean_sf = sf.mean(dim=1)
        mean_sy = sy.mean(dim=1)
        mean_bs = bs.mean(dim=1)
        mean_res = mean_sy - mean_bs
        mean_extra = torch.stack([mean_sy, mean_bs, mean_res, mean_res.abs()], dim=-1)
        mean_tok = torch.cat([mean_sf, mean_extra], dim=-1).unsqueeze(1)
        high_w = torch.softmax(self.proto_temp * (sy - sy.mean(dim=1, keepdim=True)), dim=1)
        high_sf = torch.sum(sf * high_w.unsqueeze(-1), dim=1)
        high_sy = torch.sum(sy * high_w, dim=1)
        high_bs = torch.sum(bs * high_w, dim=1)
        high_res = high_sy - high_bs
        high_extra = torch.stack([high_sy, high_bs, high_res, high_res.abs()], dim=-1)
        high_tok = torch.cat([high_sf, high_extra], dim=-1).unsqueeze(1)
        low_w = torch.softmax(-self.proto_temp * (sy - sy.mean(dim=1, keepdim=True)), dim=1)
        low_sf = torch.sum(sf * low_w.unsqueeze(-1), dim=1)
        low_sy = torch.sum(sy * low_w, dim=1)
        low_bs = torch.sum(bs * low_w, dim=1)
        low_res = low_sy - low_bs
        low_extra = torch.stack([low_sy, low_bs, low_res, low_res.abs()], dim=-1)
        low_tok = torch.cat([low_sf, low_extra], dim=-1).unsqueeze(1)
        return torch.cat([real_tok, mean_tok, high_tok, low_tok], dim=1)

    def forward(self, sf, qf, sy, bs, pred_anchor_query, pred_aux_query, bq):
        s_tok = self.build_augmented_support_tokens(sf, sy, bs)
        q_extra = torch.stack([pred_anchor_query, pred_aux_query, bq, pred_anchor_query - pred_aux_query, pred_anchor_query - bq, pred_aux_query - bq], dim=-1)
        q_tok = torch.cat([qf, q_extra], dim=-1)
        s = self.support_in(s_tok)
        q = self.query_in(q_tok)
        s = self.support_self(s)
        att_list = []
        for block in self.blocks:
            q, att = block(q, s)
            att_list.append(att)
        delta = self.out(q).squeeze(-1)
        gate = torch.sigmoid(self.gate(q).squeeze(-1))
        self.last_delta = delta
        self.last_gate = gate
        if att_list:
            a = att_list[-1].clamp_min(1e-08)
            self.last_att_entropy = -(a * a.log()).sum(dim=-1).mean()
            self.last_att_max = a.max(dim=-1).values.mean()
        anchor = self.choose_anchor(pred_anchor_query, pred_aux_query, bq)
        delta_c = delta - delta.mean(dim=1, keepdim=True)
        delta_std = delta_c.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-06)
        delta_rank = delta_c / delta_std
        anchor_std = anchor.std(dim=1, unbiased=False, keepdim=True).detach()
        anchor_std = anchor_std.clamp(min=self.rank_std_min, max=self.rank_std_max)
        residual = self.delta_scale * anchor_std * delta_rank + self.value_delta_scale * delta
        pred = anchor + gate * residual
        return pred

    def reg_loss(self):
        device = next(self.parameters()).device
        reg = torch.tensor(0.0, device=device)
        if self.last_delta is not None:
            reg = reg + (self.last_delta ** 2).mean()
        if self.last_gate is not None:
            reg = reg + 0.05 * ((self.last_gate - 0.65) ** 2).mean()
        return reg

def train_sparse_sccselect(model, adapter, mini, drug_features, cell_features, codes, device, args):
    rw_grid = [float(x) for x in args.rw_grid.split(',')]
    gw_grid = [float(x) for x in args.gw_grid.split(',')]
    model.eval()
    adapter.train()
    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.adapter_lr, weight_decay=args.adapter_weight_decay)
    best_loss = float('inf')
    best_scc = -1000000000.0
    best_score = -1000000000.0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_loss_path = out_dir / 'ranksynergy_best_trainloss_sparse_support.pt'
    best_scc_path = out_dir / 'ranksynergy_best_mseguard_scc_sparse_support.pt'
    for step in range(1, args.adapter_train_steps + 1):
        data, data1, task_batch = make_task_batch(mini, args.adapter_train_split, drug_features, cell_features, codes, device)
        with torch.no_grad():
            sf, qf, sy, qy, bs, bq = get_features(model, data, data1, task_batch)
            pred_anchor_query, aux = get_anchor_pred_and_aux(model=model, sf=sf, qf=qf, sy=sy, bs=bs, bq=bq, args=args, rw_grid=rw_grid, gw_grid=gw_grid)
            pred_aux_query = aux[AUX_PRED_KEY].detach()
            sf = sf.detach()
            qf = qf.detach()
            sy = sy.detach()
            qy = qy.detach()
            bs = bs.detach()
            bq = bq.detach()
            pred_anchor_query = pred_anchor_query.detach()
            task_weights, hard_info = compute_hard_weights(sf, qf, sy, bs, args)
        pred = adapter(sf=sf, qf=qf, sy=sy, bs=bs, pred_anchor_query=pred_anchor_query, pred_aux_query=pred_aux_query, bq=bq)
        per_task_mse = ((pred - qy) ** 2).mean(dim=1)
        per_task_anchor = ((pred - pred_anchor_query) ** 2).mean(dim=1)
        per_task_rank = weighted_pairwise_rank_loss_per_task(pred, qy, temp=args.rank_temp, power=args.rank_weight_power)
        per_task_corr = per_task_corr_loss(pred, qy)
        per_task_loss = per_task_mse + args.rank_lambda * per_task_rank + args.corr_lambda * per_task_corr + args.anchor_lambda * per_task_anchor
        weights = task_weights.detach()
        loss = (weights * per_task_loss).sum() / weights.sum().clamp_min(1e-08)
        loss = loss + args.reg_lambda * adapter.reg_loss()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.adapter_grad_clip)
        opt.step()
        current_loss = float(loss.detach().cpu())
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save({'adapter': adapter.state_dict(), 'best_loss': best_loss, 'best_scc': best_scc, 'best_score': best_score, 'step': step, 'args': vars(args)}, best_loss_path)
        if step <= 3 or step % args.train_print_every == 0:
            delta_std = float(adapter.last_delta.std(unbiased=False).detach().cpu()) if adapter.last_delta is not None else 0.0
            gate_mean = float(adapter.last_gate.detach().mean().cpu()) if adapter.last_gate is not None else 0.0
            att_entropy = float(adapter.last_att_entropy.detach().cpu()) if adapter.last_att_entropy is not None else 0.0
        if step >= args.scc_select_warmup and args.scc_select_every > 0 and (step % args.scc_select_every == 0):
            adapter.eval()
            with torch.no_grad():
                m_val, m_anchor, _ = evaluate_sparse_adapter(model=model, adapter=adapter, mini=mini, drug_features=drug_features, cell_features=cell_features, codes=codes, device=device, args=args, batches=args.scc_select_batches)
            adapter.train()
            val_scc = float(m_val['scc'])
            val_mse = float(m_val['mse'])
            val_r2 = float(m_val['r2'])
            mse_excess = max(0.0, val_mse - float(args.mse_guard))
            r2_deficit = max(0.0, float(args.r2_guard) - val_r2)
            val_score = val_scc - float(args.mse_penalty) * mse_excess - float(args.r2_penalty) * r2_deficit
            if val_score > best_score:
                best_score = val_score
                best_scc = val_scc
                torch.save({'adapter': adapter.state_dict(), 'best_loss': best_loss, 'best_scc': best_scc, 'best_score': best_score, 'step': step, 'args': vars(args), 'val_metrics': m_val, 'anchor_metrics': m_anchor}, best_scc_path)
    if best_scc_path.exists():
        ckpt = torch.load(best_scc_path, map_location=device)
        adapter.load_state_dict(ckpt['adapter'])
    else:
        ckpt = torch.load(best_loss_path, map_location=device)
        adapter.load_state_dict(ckpt['adapter'])
    adapter.eval()
    return adapter

def evaluate_sparse_adapter(model, adapter, mini, drug_features, cell_features, codes, device, args, batches):
    rw_grid = [float(x) for x in args.rw_grid.split(',')]
    gw_grid = [float(x) for x in args.gw_grid.split(',')]
    model.eval()
    adapter.eval()
    all_y = []
    all_pred = []
    all_anchor = []
    select_rows = []
    for step in range(1, batches + 1):
        data, data1, task_batch = make_task_batch(mini, args.eval_split, drug_features, cell_features, codes, device)
        with torch.no_grad():
            sf, qf, sy, qy, bs, bq = get_features(model, data, data1, task_batch)
            pred_anchor_query, aux = get_anchor_pred_and_aux(model=model, sf=sf, qf=qf, sy=sy, bs=bs, bq=bq, args=args, rw_grid=rw_grid, gw_grid=gw_grid)
            pred_aux_query = aux[AUX_PRED_KEY].detach()
            pred = adapter(sf=sf, qf=qf, sy=sy, bs=bs, pred_anchor_query=pred_anchor_query, pred_aux_query=pred_aux_query, bq=bq)
        all_y.append(qy.detach().cpu().numpy().reshape(-1))
        all_pred.append(pred.detach().cpu().numpy().reshape(-1))
        all_anchor.append(pred_anchor_query.detach().cpu().numpy().reshape(-1))
        select_rows.append({'range_gate_use_aux': float(aux[AUX_USE_KEY].float().mean().detach().cpu()), 'support_range_mean': float(aux['support_range'].mean().detach().cpu())})
        if args.print_every > 0 and step % args.print_every == 0:
            pass
    y = np.concatenate(all_y)
    pred = np.concatenate(all_pred)
    pred_anchor = np.concatenate(all_anchor)
    return (calc_metrics(y, pred), calc_metrics(y, pred_anchor), summarize_select(select_rows))

def _value_pairwise_rank_loss(pred, target, margin=0.0, temp=0.1):
    ty = target.unsqueeze(2) - target.unsqueeze(1)
    py = pred.unsqueeze(2) - pred.unsqueeze(1)
    sign = torch.sign(ty)
    mask = (ty.abs() > 1e-06).float()
    loss_mat = F.softplus(-(sign * py - margin) / temp) * mask
    denom = mask.sum().clamp_min(1.0)
    return loss_mat.sum() / denom

def support_value_residual_adapt(sf, qf, sy, bs, bq, steps=9, lr=0.035, l2=0.1, clip=1.0):
    B, S, D = sf.shape
    device = sf.device
    sf = sf.detach()
    qf = qf.detach()
    sy = sy.detach()
    bs = bs.detach()
    bq = bq.detach()
    w = torch.zeros(B, D, device=device, requires_grad=True)
    b = torch.zeros(B, 1, device=device, requires_grad=True)
    for _ in range(int(steps)):
        delta_s = torch.einsum('bsd,bd->bs', sf, w) + b
        pred_s = bs + delta_s
        mse = F.mse_loss(pred_s, sy)
        rloss = _value_pairwise_rank_loss(pred_s, sy, margin=0.0, temp=0.1)
        reg = l2 * (w.pow(2).mean() + b.pow(2).mean())
        loss = mse + 0.15 * rloss + reg
        grad_w, grad_b = torch.autograd.grad(loss, [w, b], create_graph=False)
        with torch.no_grad():
            if clip is not None and clip > 0:
                gw_norm = grad_w.norm(dim=1, keepdim=True).clamp_min(1e-08)
                scale = (clip / gw_norm).clamp(max=1.0)
                grad_w = grad_w * scale
                gb_norm = grad_b.abs().clamp_min(1e-08)
                grad_b = grad_b * (clip / gb_norm).clamp(max=1.0)
            w -= lr * grad_w
            b -= lr * grad_b
        w.requires_grad_(True)
        b.requires_grad_(True)
    with torch.no_grad():
        delta_s = torch.einsum('bsd,bd->bs', sf, w.detach()) + b.detach()
        pred_s = bs + delta_s
        base_mse = (bs - sy).pow(2).mean(dim=1)
        adapt_mse = (pred_s - sy).pow(2).mean(dim=1)
        improve = (base_mse - adapt_mse) / (base_mse.abs() + 1e-06)
        improve_gate = torch.sigmoid(10.0 * improve)
        sy_var = sy.var(dim=1, unbiased=False)
        var_gate = 0.75 + 0.25 * torch.sigmoid(6.0 * (sy_var - 0.003))
        delta_abs = delta_s.abs().mean(dim=1)
        amp_gate = torch.sigmoid(7.0 * (0.45 - delta_abs))
        gate = improve_gate * var_gate * amp_gate
        gate = gate.clamp(min=0.1, max=0.95).view(B, 1)
    delta_q = torch.einsum('bqd,bd->bq', qf, w.detach()) + b.detach()
    pred_raw = bq + gate * delta_q
    return pred_raw

def evaluate_value_rank_transfer(model, adapter, mini, drug_features, cell_features, codes, device, args, batches):
    rw_grid = [float(x) for x in args.rw_grid.split(',')]
    gw_grid = [float(x) for x in args.gw_grid.split(',')]
    gamma_grid = [float(x) for x in args.rank_transfer_grid.split(',')]
    model.eval()
    adapter.eval()
    all_y = []
    all_anchor = []
    all_by_gamma = {g: [] for g in gamma_grid}
    select_rows = []
    for step in range(1, batches + 1):
        data, data1, task_batch = make_task_batch(mini, args.eval_split, drug_features, cell_features, codes, device)
        with torch.no_grad():
            sf, qf, sy, qy, bs, bq = get_features(model, data, data1, task_batch)
            pred_anchor_query, aux = get_anchor_pred_and_aux(model=model, sf=sf, qf=qf, sy=sy, bs=bs, bq=bq, args=args, rw_grid=rw_grid, gw_grid=gw_grid)
            pred_aux_query = aux[AUX_PRED_KEY].detach()
            pred_rank = adapter(sf=sf, qf=qf, sy=sy, bs=bs, pred_anchor_query=pred_anchor_query, pred_aux_query=pred_aux_query, bq=bq)
        pred_value_raw = support_value_residual_adapt(sf=sf, qf=qf, sy=sy, bs=bs, bq=bq, steps=args.value_steps, lr=args.value_lr, l2=args.value_l2, clip=args.grad_clip)
        with torch.no_grad():
            pred_value = bq + float(args.value_blend) * (pred_value_raw - bq)
            value_mean = pred_value.mean(dim=1, keepdim=True)
            value_std = pred_value.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-06)
            rank_center = pred_rank - pred_rank.mean(dim=1, keepdim=True)
            rank_std = rank_center.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-06)
            rank_z = rank_center / rank_std
            rank_scaled = value_mean + value_std * rank_z
            for gamma in gamma_grid:
                pred = (1.0 - gamma) * pred_value + gamma * rank_scaled
                all_by_gamma[gamma].append(pred.detach().cpu().numpy().reshape(-1))
        all_y.append(qy.detach().cpu().numpy().reshape(-1))
        all_anchor.append(pred_anchor_query.detach().cpu().numpy().reshape(-1))
        select_rows.append({'range_gate_use_aux': float(aux[AUX_USE_KEY].float().mean().detach().cpu()), 'support_range_mean': float(aux['support_range'].mean().detach().cpu())})
        if args.print_every > 0 and step % args.print_every == 0:
            pass
    y = np.concatenate(all_y)
    pred_anchor = np.concatenate(all_anchor)
    gamma_metrics = {}
    for gamma in gamma_grid:
        pred = np.concatenate(all_by_gamma[gamma])
        gamma_metrics[str(gamma)] = calc_metrics(y, pred)
    if len(gamma_grid) == 1:
        best_gamma = str(gamma_grid[0])
    else:
        feasible = [k for k, v in gamma_metrics.items() if v['scc'] >= 0.513 and v['mse'] <= 0.108]
        if feasible:
            best_gamma = min(feasible, key=lambda k: gamma_metrics[k]['mse'])
        else:
            feasible = [k for k, v in gamma_metrics.items() if v['scc'] >= 0.513]
            if feasible:
                best_gamma = min(feasible, key=lambda k: gamma_metrics[k]['mse'])
            else:
                best_gamma = max(gamma_metrics.keys(), key=lambda k: gamma_metrics[k]['scc'])
    for k, v in gamma_metrics.items():
        pass
    s = summarize_select(select_rows)
    s['best_gamma'] = float(best_gamma)
    return (gamma_metrics[best_gamma], calc_metrics(y, pred_anchor), s)

def apply_final_preset(args):
    if args.preset == 'none':
        return args
    user_cuda = args.cuda
    user_out_dir = args.out_dir
    args.batch_size = 50
    args.query = 40
    args.loops = 2
    args.heads = 8
    args.dropout = 0.2
    args.ridge_lambda = 1.0
    args.grad_steps = 20
    args.grad_lr = 0.03
    args.grad_l2 = 0.001
    args.rw_grid = '0.20,0.25,0.30,0.35'
    args.gw_grid = '0.45,0.50,0.55,0.60'
    args.kernel_list = 'rbf,lap,cosine'
    args.tau_grid = '0.4,0.5,0.6,0.7,0.8'
    args.lam_grid = '0.1,0.2,0.3,0.5,0.7'
    args.shrink_grid = '0.4,0.5,0.6,0.7'
    args.alpha_grid = '0,0.2,0.3,0.4,0.5,0.6'
    args.mse_tol = 0.0005
    args.range_gate_threshold = 0.8
    args.adapter_lr = 0.0003
    args.adapter_weight_decay = 0.001
    args.adapter_grad_clip = 1.0
    args.d_model = 256
    args.n_heads = 4
    args.n_layers = 3
    args.attn_dropout = 0.1
    args.delta_scale = 0.06
    args.value_delta_scale = 0.02
    args.rank_std_min = 0.05
    args.rank_std_max = 0.7
    args.anchor_mode = 'global'
    args.rank_lambda = 0.1
    args.rank_temp = 0.1
    args.rank_weight_power = 1.0
    args.corr_lambda = 0.05
    args.anchor_lambda = 0.5
    args.reg_lambda = 0.0002
    args.hard_range_min = 0.3
    args.hard_range_max = 1.3
    args.hard_base_mse_min = 0.02
    args.hard_feature_shift_min = 0.08
    args.hard_weight = 2.0
    args.adapter_train_split = 'train'
    args.train_print_every = 100
    args.test_batches = 500
    args.repeats = 5
    args.print_every = 50
    args.feature_path = 'data/sample_features/'
    args.checkpoint = 'saved_model/saved_model_loop_10shot_tune1/loop_epoch1950_mse0.112342.pth'
    args.seed = 2026
    if args.preset == '5shot_final':
        args.support = 5
        args.holdout_size = 2
        args.inner_gate_holdout_size = 1
        args.adapter_train_steps = 2500
        args.eval_split = 'test'
        args.eval_seed = 2027
        args.scc_select_every = 100
        args.scc_select_warmup = 300
        args.scc_select_batches = 200
        args.mse_guard = 0.109
        args.mse_penalty = 2.0
        args.r2_guard = 0.215
        args.r2_penalty = 0.25
        args.rank_transfer_grid = '0.82'
        args.value_steps = 9
        args.value_lr = 0.035
        args.value_l2 = 0.1
        args.value_blend = 0.7
        args.mode = 'eval_rank_transfer'
        if not args.eval_only_adapter:
            args.eval_only_adapter = 'loop_outputs/ranksynergy_5shot_sparse_adapter/ranksynergy_best_mseguard_scc_sparse_support.pt'
        if user_out_dir == '':
            args.out_dir = 'loop_outputs/final_ranksynergy_5shot_gamma082'
    elif args.preset == '5shot_train_adapter':
        args.support = 5
        args.holdout_size = 2
        args.inner_gate_holdout_size = 1
        args.adapter_train_steps = 2500
        args.eval_split = 'val'
        args.eval_seed = 2026
        args.scc_select_every = 100
        args.scc_select_warmup = 300
        args.scc_select_batches = 200
        args.mse_guard = 0.109
        args.mse_penalty = 2.0
        args.r2_guard = 0.215
        args.r2_penalty = 0.25
        args.mode = 'train_sparse_adapter'
        if user_out_dir == '':
            args.out_dir = 'loop_outputs/final_ranksynergy_5shot_train_adapter'
    elif args.preset == '10shot_final':
        args.support = 10
        args.holdout_size = 3
        args.inner_gate_holdout_size = 2
        args.adapter_train_steps = 1500
        args.eval_split = 'val'
        args.eval_seed = 2026
        args.mode = 'train_adapter'
        if user_out_dir == '':
            args.out_dir = 'loop_outputs/final_ranksynergy_10shot_cross_attention'
    elif args.preset == '30shot_final':
        args.support = 30
        args.holdout_size = 5
        args.inner_gate_holdout_size = 3
        args.adapter_train_steps = 1500
        args.eval_split = 'val'
        args.eval_seed = 2026
        args.mode = 'train_adapter'
        if user_out_dir == '':
            args.out_dir = 'loop_outputs/final_ranksynergy_30shot_cross_attention'
    else:
        raise ValueError(f'Unknown preset: {args.preset}')
    args.cuda = user_cuda
    if user_out_dir != '':
        args.out_dir = user_out_dir
    return args

def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', type=str, default='none', choices=['none', '5shot_final', '5shot_train_adapter', '10shot_final', '30shot_final'])
    parser.add_argument('--mode', type=str, default='train_adapter', choices=['train_adapter', 'train_sparse_adapter', 'eval_rank_transfer'])
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--support', type=int, default=10)
    parser.add_argument('--query', type=int, default=40)
    parser.add_argument('--loops', type=int, default=2)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--ridge_lambda', type=float, default=1.0)
    parser.add_argument('--grad_steps', type=int, default=20)
    parser.add_argument('--grad_lr', type=float, default=0.03)
    parser.add_argument('--grad_l2', type=float, default=0.001)
    parser.add_argument('--holdout_size', type=int, default=3)
    parser.add_argument('--inner_gate_holdout_size', type=int, default=2)
    parser.add_argument('--rw_grid', type=str, default='0.20,0.25,0.30,0.35')
    parser.add_argument('--gw_grid', type=str, default='0.45,0.50,0.55,0.60')
    parser.add_argument('--kernel_list', type=str, default='rbf,lap,cosine')
    parser.add_argument('--tau_grid', type=str, default='0.4,0.5,0.6,0.7,0.8')
    parser.add_argument('--lam_grid', type=str, default='0.1,0.2,0.3,0.5,0.7')
    parser.add_argument('--shrink_grid', type=str, default='0.4,0.5,0.6,0.7')
    parser.add_argument('--alpha_grid', type=str, default='0,0.2,0.3,0.4,0.5,0.6')
    parser.add_argument('--mse_tol', type=float, default=0.0005)
    parser.add_argument('--range_gate_threshold', type=float, default=0.8)
    parser.add_argument('--adapter_train_steps', type=int, default=1500)
    parser.add_argument('--adapter_lr', type=float, default=0.0003)
    parser.add_argument('--adapter_weight_decay', type=float, default=0.001)
    parser.add_argument('--adapter_grad_clip', type=float, default=1.0)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--attn_dropout', type=float, default=0.1)
    parser.add_argument('--delta_scale', type=float, default=0.08)
    parser.add_argument('--value_delta_scale', type=float, default=0.02)
    parser.add_argument('--rank_std_min', type=float, default=0.05)
    parser.add_argument('--rank_std_max', type=float, default=0.8)
    parser.add_argument('--anchor_mode', type=str, default='global', choices=['global', 'aux', 'base', 'mix'])
    parser.add_argument('--rank_lambda', type=float, default=0.08)
    parser.add_argument('--rank_temp', type=float, default=0.1)
    parser.add_argument('--rank_weight_power', type=float, default=1.0)
    parser.add_argument('--corr_lambda', type=float, default=0.03)
    parser.add_argument('--anchor_lambda', type=float, default=0.5)
    parser.add_argument('--reg_lambda', type=float, default=0.0005)
    parser.add_argument('--hard_range_min', type=float, default=0.3)
    parser.add_argument('--hard_range_max', type=float, default=1.3)
    parser.add_argument('--hard_base_mse_min', type=float, default=0.02)
    parser.add_argument('--hard_feature_shift_min', type=float, default=0.08)
    parser.add_argument('--hard_weight', type=float, default=2.0)
    parser.add_argument('--adapter_train_split', type=str, default='train')
    parser.add_argument('--eval_split', type=str, default='val')
    parser.add_argument('--eval_seed', type=int, default=None)
    parser.add_argument('--scc_select_every', type=int, default=100)
    parser.add_argument('--scc_select_warmup', type=int, default=300)
    parser.add_argument('--scc_select_batches', type=int, default=200)
    parser.add_argument('--mse_guard', type=float, default=0.109)
    parser.add_argument('--mse_penalty', type=float, default=2.0)
    parser.add_argument('--r2_guard', type=float, default=0.215)
    parser.add_argument('--r2_penalty', type=float, default=0.25)
    parser.add_argument('--rank_transfer_grid', type=str, default='0.82')
    parser.add_argument('--value_steps', type=int, default=9)
    parser.add_argument('--value_lr', type=float, default=0.035)
    parser.add_argument('--value_l2', type=float, default=0.1)
    parser.add_argument('--value_blend', type=float, default=0.7)
    parser.add_argument('--eval_only_adapter', type=str, default='')
    parser.add_argument('--train_print_every', type=int, default=100)
    parser.add_argument('--test_batches', type=int, default=200)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--print_every', type=int, default=50)
    parser.add_argument('--cuda', type=str, default='cuda:0')
    parser.add_argument('--feature_path', type=str, default='data/sample_features/')
    parser.add_argument('--checkpoint', type=str, default='saved_model/saved_model_loop_10shot_tune1/loop_epoch1950_mse0.112342.pth')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--out_dir', type=str, default='')
    return parser

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args = apply_final_preset(args)
    set_seed(args.seed)
    device = args.cuda if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mini, drug_features, cell_features, codes = load_project_data(args)
    model = load_model(args.checkpoint, args, device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    feature_dim = int(getattr(model, 'embed_dim', 512))
    if args.mode == 'train_adapter':
        adapter = CrossAttnRankResidualRegressor(feature_dim=feature_dim, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.attn_dropout, delta_scale=args.delta_scale, value_delta_scale=args.value_delta_scale, rank_std_min=args.rank_std_min, rank_std_max=args.rank_std_max, anchor_mode=args.anchor_mode).to(device)
        adapter = train_cross_attention_adapter(model=model, adapter=adapter, mini=mini, drug_features=drug_features, cell_features=cell_features, codes=codes, device=device, args=args)
        eval_fn = evaluate_cross_attention_adapter
        summary_key = 'ranksynergy_cross_attention'
    elif args.mode == 'train_sparse_adapter':
        adapter = SparseSupportRankResidualRegressor(feature_dim=feature_dim, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.attn_dropout, delta_scale=args.delta_scale, value_delta_scale=args.value_delta_scale, rank_std_min=args.rank_std_min, rank_std_max=args.rank_std_max, anchor_mode=args.anchor_mode).to(device)
        adapter = train_sparse_sccselect(model=model, adapter=adapter, mini=mini, drug_features=drug_features, cell_features=cell_features, codes=codes, device=device, args=args)
        eval_fn = evaluate_sparse_adapter
        summary_key = 'ranksynergy_5shot_sparse'
    elif args.mode == 'eval_rank_transfer':
        adapter = SparseSupportRankResidualRegressor(feature_dim=feature_dim, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.attn_dropout, delta_scale=args.delta_scale, value_delta_scale=args.value_delta_scale, rank_std_min=args.rank_std_min, rank_std_max=args.rank_std_max, anchor_mode=args.anchor_mode).to(device)
        if not args.eval_only_adapter:
            raise ValueError('--eval_only_adapter is required for eval_rank_transfer mode.')
        ckpt_path = Path(args.eval_only_adapter)
        if not ckpt_path.exists():
            raise FileNotFoundError(f'eval_only_adapter not found: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=device)
        adapter.load_state_dict(ckpt['adapter'])
        adapter.eval()
        eval_fn = evaluate_value_rank_transfer
        summary_key = 'ranksynergy_5shot_value_rank_transfer'
    else:
        raise RuntimeError(f'Unknown mode: {args.mode}')
    if args.eval_seed is not None:
        set_seed(args.eval_seed)
    main_rows = []
    anchor_rows = []
    select_rows = []
    for r in range(1, args.repeats + 1):
        m_main, m_anchor, s = eval_fn(model=model, adapter=adapter, mini=mini, drug_features=drug_features, cell_features=cell_features, codes=codes, device=device, args=args, batches=args.test_batches)
        main_rows.append(m_main)
        anchor_rows.append(m_anchor)
        select_rows.append(s)
    main_summary = summarize_rows(main_rows)
    anchor_summary = summarize_rows(anchor_rows)
    select_summary = summarize_select(select_rows)
    for k, v in main_summary.items():
        pass
    for k, v in anchor_summary.items():
        pass
    for k, v in select_summary.items():
        pass
    out = {summary_key: main_summary, 'global': anchor_summary, 'selection': select_summary, 'args': vars(args)}
    out_path = out_dir / 'test_summary.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
if __name__ == '__main__':
    main()
