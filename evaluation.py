import csv
import logging
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from training import CERTDataset

logger = logging.getLogger(__name__)

def evaluate_and_extract_z_optimized(
    model, test_pkl, batch_size=128, device="cuda", weights=None, vocab_size=8,
    low_score_anomaly_dbscan_eps=0.15, low_score_anomaly_dbscan_min_samples=200,
    tta_ema_alpha=0.02, tta_radius_tolerance=1.4, proto_penalty_weight=20.0,
    enable_tta=True, baseline_f1_threshold=1.5,
    output_dir=None
):
    # 基础初始化与数据加载逻辑
    from training import CERTDataset
    dataset = CERTDataset(test_pkl)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    if weights is None: weights = {"num": 1.0, "ctx": 1.0, "seq": 0.1}
    model.eval()
    
    z_list, err_list, labels_list, user_dates = [], [], [], []
    raw_num_list, raw_ctx_dyn_list, raw_ctx_stat_list, raw_seq_list = [], [], [], []
    
    with torch.no_grad():
        for batch in loader:
            num, seq = batch["num"].to(device), batch["seq"].to(device)
            ctx_dyn, ctx_stat = batch["ctx_dynamic"].to(device), batch["ctx_static"].to(device)
            
            z, r_num, r_ctx_dyn, r_seq, min_dist, _ = model(num, ctx_dyn, ctx_stat, seq)
            
            # 计算重构误差
            e_num = torch.mean((r_num - num) ** 2, dim=1)
            e_ctx = torch.mean((r_ctx_dyn - ctx_dyn) ** 2, dim=1)
            ce = nn.CrossEntropyLoss(ignore_index=0, reduction="none")(r_seq.view(-1, vocab_size), seq.view(-1))
            e_seq = ce.view(num.size(0), -1).mean(dim=1)
            
            recon_err = weights["num"] * e_num + weights["ctx"] * e_ctx + weights["seq"] * e_seq
            
            z_list.append(z.cpu().numpy())
            err_list.append(recon_err.cpu().numpy())
            labels_list.extend(batch["label"].numpy())
            user_dates.extend(batch["user_date"])
            
            raw_num_list.append(num.cpu())
            raw_ctx_dyn_list.append(ctx_dyn.cpu())
            raw_ctx_stat_list.append(ctx_stat.cpu())
            raw_seq_list.append(seq.cpu())
            
    Z = np.concatenate(z_list, axis=0)
    score_recon = np.concatenate(err_list, axis=0)
    labels = np.array(labels_list)
    
    # 动态为模型初始化全局联合半径
    if not hasattr(model, 'unified_radii') or model.unified_radii is None:
        model.unified_radii = torch.ones(model.proto_layer.prototypes.size(0), device=device) * 0.35

    # 优化 2：基于中位数与标准差的稳健动态防火墙
    if not hasattr(model, 'ema_recon_median'):
        model.ema_recon_median = np.median(score_recon)
        model.ema_recon_std = np.std(score_recon)
    
    # 防火墙阈值：重构误差大于 (中位数 + 1.5 * 标准差) 的，直接剥夺演进资格
    dynamic_firewall_thresh = model.ema_recon_median + 1.5 * model.ema_recon_std
    logger.info(f"   [Dynamic Firewall] Median={model.ema_recon_median:.4f}, Std={model.ema_recon_std:.4f}, Thresh={dynamic_firewall_thresh:.4f}")

    final_anomaly_scores = score_recon.copy()
    new_cluster_data = None
    
    if enable_tta:
        Z_tensor = torch.tensor(Z, device=device)
        # 获取最新的原型拓扑
        protos_norm = torch.nn.functional.normalize(model.proto_layer.prototypes.data, p=2, dim=-1)
        Z_norm = torch.nn.functional.normalize(Z_tensor, p=2, dim=1)
        
        # 使用与训练一致的缩放余弦距离
        cosine_sim = torch.matmul(Z_norm, protos_norm.T)
        dists_matrix = (1.0 - cosine_sim) / 0.1 
        min_dists, hit_ids = torch.min(dists_matrix, dim=1)
        
        radii_thresh = model.unified_radii[hit_ids].cpu().numpy() * tta_radius_tolerance
        
        # 【动态防火墙】：使用当前批次重构误差的 85% 分位数代替硬编码，防止高频攻击通过
        dynamic_recon_thresh = np.percentile(score_recon, 85)
        safe_recon_mask = score_recon < min(baseline_f1_threshold, dynamic_recon_thresh)
        
        is_drift_candidate = (min_dists.cpu().numpy() > radii_thresh) & safe_recon_mask
        is_inlier = (min_dists.cpu().numpy() <= radii_thresh)
        
        # 【防中毒机制】：置信度加权的 EMA 顺应模式更新
        for k in range(protos_norm.size(0)):
            k_mask = (hit_ids.cpu().numpy() == k) & is_inlier
            if np.any(k_mask):
                k_z = Z_tensor[torch.from_numpy(k_mask).to(device)]
                k_dists = min_dists[torch.from_numpy(k_mask).to(device)]
                
                # 置信度权重：距离越近，权重越接近 1；距离越靠近边界，权重越接近 0
                confidence = torch.exp(-k_dists).unsqueeze(1)
                effective_alpha = tta_ema_alpha * confidence
                
                # 动态微调该原型
                updated_protos = (1 - effective_alpha) * protos_norm[k] + effective_alpha * k_z
                model.proto_layer.prototypes.data[k] = torch.mean(updated_protos, dim=0)

        # 概念漂移发现（DBSCAN）
        drift_indices = np.where(is_drift_candidate)[0]
        if len(drift_indices) >= low_score_anomaly_dbscan_min_samples:
            # 漂移聚类应该用原生的表征
            dbscan = DBSCAN(eps=low_score_anomaly_dbscan_eps, min_samples=low_score_anomaly_dbscan_min_samples, metric="cosine")
            res_labels = dbscan.fit_predict(Z_norm[drift_indices].cpu().numpy())
            
            unique_clusters = set(res_labels) - {-1}
            valid_new_indices = []
            
            for c in unique_clusters:
                c_mask = (res_labels == c)
                curr_indices = drift_indices[c_mask]
                
                # 双重校验：该新簇的平均重构误差必须低于安全线
                if np.mean(score_recon[curr_indices]) > dynamic_recon_thresh:
                    continue 
                
                valid_new_indices.extend(curr_indices.tolist())
                Z_c = Z_tensor[curr_indices]
                centroid = torch.nn.functional.normalize(torch.mean(Z_c, dim=0, keepdim=True), p=2, dim=-1)
                
                # 设定新原型的有效管控半径
                cluster_dists = (1.0 - torch.matmul(torch.nn.functional.normalize(Z_c, p=2, dim=-1), centroid.T)) / 0.1
                r_95 = max(torch.quantile(cluster_dists.view(-1), 0.95).item(), 0.15)
                
                model.proto_layer.prototypes = nn.Parameter(torch.cat([model.proto_layer.prototypes.data, centroid], dim=0))
                model.unified_radii = torch.cat([model.unified_radii, torch.tensor([r_95], device=device)], dim=0)

            # 更新完毕后，重新评估所有样本的最终距离与 Margin
            protos_norm_final = torch.nn.functional.normalize(model.proto_layer.prototypes.data, p=2, dim=-1)
            cosine_sim_final = torch.matmul(Z_norm, protos_norm_final.T)
            dists_final = ((1.0 - cosine_sim_final) / 0.1).cpu().numpy()
            
            f_min_dist = np.min(dists_final, axis=1)
            f_hit_id = np.argmin(dists_final, axis=1)
            
            final_radii = model.unified_radii.cpu().numpy()
            margin = np.maximum(f_min_dist - (final_radii[f_hit_id] * tta_radius_tolerance), 0.0)
            
            final_anomaly_scores = score_recon + (proto_penalty_weight * margin)
            
            if len(valid_new_indices) > 0:
                valid_new_indices = np.unique(valid_new_indices)
                new_cluster_data = {
                    "num": torch.cat(raw_num_list, dim=0)[valid_new_indices],
                    "ctx_dynamic": torch.cat(raw_ctx_dyn_list, dim=0)[valid_new_indices],
                    "ctx_static": torch.cat(raw_ctx_stat_list, dim=0)[valid_new_indices],
                    "seq": torch.cat(raw_seq_list, dim=0)[valid_new_indices]
                }

        # 监控原型半径膨胀率
        radii_mean = model.unified_radii.mean().item()
        radii_max = model.unified_radii.max().item()
        logger.info(f"   [Radii Monitor] Mean={radii_mean:.4f}, Max={radii_max:.4f}")

    # 落表：重构误差与原型距离的散点分布
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        scatter_path = os.path.join(output_dir, f"{os.path.basename(test_pkl)}_scatter_recon_dist.csv")
        with open(scatter_path, "w", encoding="utf-8") as f:
            f.write("score_recon,min_dist,label\n")
            for sr, md, lb in zip(score_recon, min_dists_np, labels):
                f.write(f"{sr:.6f},{md:.6f},{lb}\n")
        logger.info(f"   [Scatter Plot Data] Saved to {scatter_path}")

    # 指标统计与返回
    auc = roc_auc_score(labels, final_anomaly_scores) if len(set(labels)) > 1 else 0.5
    pr_auc = average_precision_score(labels, final_anomaly_scores) if len(set(labels)) > 1 else 0.0
    
    return {
        "scores": final_anomaly_scores, "labels": labels, "auc": auc, "pr_auc": pr_auc,
        "new_cluster_data": new_cluster_data, "user_dates": user_dates
    }


def precision_at_k(y_true, scores, k: int = 100) -> float:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    k = min(int(k), len(y_sorted))
    if k == 0: return 0.0
    return float(np.sum(y_sorted[:k] == 1) / k)

def precision_at_k_multi(y_true, scores, ks=(50, 100)) -> dict[int, float]:
    out = {}
    for k in ks:
        out[int(k)] = precision_at_k(y_true, scores, k=int(k))
    return out

def compute_eer(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)

def detection_rate_at_budgets(y_true, scores, budgets=(0.05, 0.10, 0.15)):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    total_pos = np.sum(y_true == 1)
    if total_pos == 0: return {float(b): 0.0 for b in budgets}
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    out = {}
    total_samples = len(y_true)
    for b in budgets:
        k = min(int(np.ceil(b * total_samples)), total_samples)
        out[float(b)] = float(np.sum(y_sorted[:k] == 1) / total_pos) if k > 0 else 0.0
    return out

def _binary_curve_from_scores(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    total_pos = int(np.sum(y_true == 1))
    total_neg = int(np.sum(y_true == 0))
    if total_pos == 0 or total_neg == 0: return None
    order = np.argsort(-scores)
    scores_sorted = scores[order]
    y_sorted = y_true[order]
    cum_tp = np.cumsum(y_sorted == 1)
    cum_fp = np.cumsum(y_sorted == 0)
    last_idx = np.where(np.diff(scores_sorted) != 0)[0]
    candidate_idx = np.concatenate([last_idx, np.array([len(scores_sorted) - 1])])
    tp = cum_tp[candidate_idx].astype(int)
    fp = cum_fp[candidate_idx].astype(int)
    fn = (total_pos - tp).astype(int)
    tn = (total_neg - fp).astype(int)
    thresholds = scores_sorted[candidate_idx].astype(float)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, total_pos, out=np.zeros_like(tp, dtype=float), where=total_pos != 0)
    fpr = np.divide(fp, total_neg, out=np.zeros_like(fp, dtype=float), where=total_neg != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision, dtype=float), where=(precision + recall) != 0)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "thresholds": thresholds, "precision": precision, "recall": recall, "fpr": fpr, "f1": f1, "total_pos": total_pos, "total_neg": total_neg}

def metrics_at_best_f1(y_true, scores):
    curve = _binary_curve_from_scores(y_true, scores)
    if curve is None: return None
    best_i = int(np.argmax(curve["f1"]))
    best_thr = float(curve["thresholds"][best_i])
    y_pred = (scores >= best_thr).astype(int)
    tn2, fp2, fn2, tp2 = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": best_thr, "tp": int(tp2), "fp": int(fp2), "tn": int(tn2), "fn": int(fn2),
        "fpr": float(fp2 / (fp2 + tn2)) if (fp2 + tn2) > 0 else 0.0,
        "tpr": float(tp2 / (tp2 + fn2)) if (tp2 + fn2) > 0 else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(curve["f1"][best_i]),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred))
    }

def fpr_at_target_recalls(y_true, scores, target_recalls=(0.8, 0.9, 0.95)):
    curve = _binary_curve_from_scores(y_true, scores)
    if curve is None: return {float(r): None for r in target_recalls}
    out = {}
    for target in target_recalls:
        idx = np.where(curve["recall"] >= float(target))[0]
        if len(idx) == 0:
            out[float(target)] = None
            continue
        best_i = int(idx[np.argmin(curve["fpr"][idx])])
        out[float(target)] = {
            "fpr": float(curve["fpr"][best_i]), "threshold": float(curve["thresholds"][best_i]),
            "precision": float(curve["precision"][best_i]), "recall": float(curve["recall"][best_i]),
            "tp": int(curve["tp"][best_i]), "fp": int(curve["fp"][best_i]), "tn": int(curve["tn"][best_i]), "fn": int(curve["fn"][best_i])
        }
    return out

def _save_pr_curve(y_true, scores, save_path: str) -> bool:
    if int(np.sum(y_true == 1)) == 0 or int(np.sum(y_true == 0)) == 0: return False
    precision, recall, _ = precision_recall_curve(y_true, scores)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR-AUC={average_precision_score(y_true, scores):.4f}", linewidth=2)
    plt.xlim(0.0, 1.0); plt.ylim(0.0, 1.0)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
    plt.grid(alpha=0.3); plt.legend(loc="best"); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    return True

def _save_score_distribution(y_true, scores, save_path: str) -> bool:
    normal_scores, anomaly_scores = scores[y_true == 0], scores[y_true == 1]
    if len(normal_scores) == 0 or len(anomaly_scores) == 0: return False
    plt.figure(figsize=(8, 6))
    plt.hist(normal_scores, bins=60, alpha=0.6, density=True, label=f"Normal (n={len(normal_scores)})")
    plt.hist(anomaly_scores, bins=60, alpha=0.6, density=True, label=f"Anomaly (n={len(anomaly_scores)})")
    plt.xlabel("Anomaly Score"); plt.ylabel("Density"); plt.title("Score Distribution Overlap")
    plt.grid(alpha=0.3); plt.legend(loc="best"); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    return True

def fallback_metrics_single_class(y_true, scores):
    y_pred = np.zeros_like(y_true, dtype=int) if np.sum(y_true == 1) == 0 else np.ones_like(y_true, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float('inf'), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "fpr": 0.0, "tpr": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "f2": 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred))
    }

def _get_multihead_prototypes_concat(model) -> np.ndarray | None:
    if not all(hasattr(model, name) for name in ("proto_layer_num", "proto_layer_dyn", "proto_layer_seq")): return None
    p_num = F.normalize(model.proto_layer_num.prototypes.data, p=2, dim=1)
    p_dyn = F.normalize(model.proto_layer_dyn.prototypes.data, p=2, dim=1)
    p_seq = F.normalize(model.proto_layer_seq.prototypes.data, p=2, dim=1)
    max_k = max(p_num.size(0), p_dyn.size(0), p_seq.size(0))
    pad_proto = lambda p: p if p.size(0) == max_k else F.pad(p, (0, 0, 0, max_k - p.size(0)), "constant", 0.0)
    return torch.cat([pad_proto(p_num), pad_proto(p_dyn), pad_proto(p_seq)], dim=1).detach().cpu().numpy()

def _save_latent_space_tsne(z_vectors: np.ndarray, labels: np.ndarray, model, save_path: str, max_samples: int = 5000) -> bool:
    prototypes = _get_multihead_prototypes_concat(model)
    if prototypes is None: return False
    num_prototypes = int(prototypes.shape[0])
    if len(z_vectors) > max_samples:
        np.random.seed(42)
        idx = np.random.choice(len(z_vectors), max_samples, replace=False)
        z_sample, label_sample = z_vectors[idx], labels[idx]
    else:
        z_sample, label_sample = z_vectors, labels
    features_2d = TSNE(n_components=2, perplexity=30, random_state=42, init="pca").fit_transform(np.vstack([z_sample, prototypes]))
    z_2d, p_2d = features_2d[:-num_prototypes], features_2d[-num_prototypes:]
    plt.figure(figsize=(12, 10))
    if np.any(label_sample == 0): plt.scatter(z_2d[label_sample == 0, 0], z_2d[label_sample == 0, 1], alpha=0.3, s=15, c="#1f77b4", label="Normal")
    if np.any(label_sample == 1): plt.scatter(z_2d[label_sample == 1, 0], z_2d[label_sample == 1, 1], alpha=0.6, s=25, c="#ff7f0e", marker="x", label="Anomaly")
    plt.scatter(p_2d[:, 0], p_2d[:, 1], marker="*", c="#d62728", s=300, edgecolor="black", label="Prototypes")
    plt.title(f"t-SNE Latent Space & Prototypes (k={num_prototypes})", fontsize=14)
    plt.legend(loc="best"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(save_path, dpi=200); plt.close()
    return True

def evaluate_and_extract_z(
    model, test_pkl: str, batch_size: int = 128, device: str | torch.device = "cuda",
    weights: dict | None = None, vocab_size: int = 8, topk_list=(50, 100),
    target_recalls=(0.8, 0.9, 0.95), budget_list=(0.05, 0.10, 0.15),
    figure_dir: str | None = None, dump_low_score_anomalies: bool = True,
    low_score_anomaly_top_ratio: float = 0.5, low_score_anomaly_top_n: int = 200,
    low_score_anomaly_dbscan_eps: float = 0.005,
    low_score_anomaly_dbscan_min_samples: int = 1000,
    tta_ema_alpha: float = 0.02,
    tta_radius_tolerance: float = 0.7,
    proto_penalty_weight: float = 15.0,
    enable_tta: bool = True,
    enable_pos_profiling: bool = True,
    extract_new_clusters: bool = True,
):
    logger.info("[%s] >>> 启动双边界分流原型演进检测 Pipeline (TTA=%s): %s", datetime.now(), enable_tta, os.path.basename(test_pkl))
    test_dataset = CERTDataset(test_pkl)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=int(batch_size), shuffle=False)
    
    if weights is None: weights = {"num": 1.0, "ctx": 2.0, "seq": 0.1}
    w_num, w_ctx, w_seq = float(weights.get("num", 1.0)), float(weights.get("ctx", 2.0)), float(weights.get("seq", 0.1))
    model.eval()

    z_vectors, err_num_all, err_ctx_all, err_seq_all = [], [], [] ,[]
    raw_num_list, raw_ctx_dyn_list, raw_ctx_stat_list, raw_seq_list = [], [], [], []
    all_labels, all_user_dates, all_indices = [], [], []
    global_idx = 0

    with torch.no_grad():
        for batch in test_loader:
            num, seq = batch["num"].to(device), batch["seq"].to(device)
            ctx_dyn, ctx_stat = batch.get("ctx_dynamic").to(device), batch.get("ctx_static").to(device)
            ctx = torch.cat([ctx_dyn, ctx_stat], dim=1)

            outputs = model(num, ctx_dyn, ctx_stat, seq)
            z, r_num, r_ctx, r_seq = outputs[0], outputs[1], outputs[2], outputs[3]

            err_num = torch.mean((r_num - num) ** 2, dim=1)
            err_ctx = torch.mean((r_ctx - ctx) ** 2, dim=1) if torch.is_tensor(r_ctx) else torch.zeros(num.size(0), device=device)
            ce_loss = nn.CrossEntropyLoss(ignore_index=0, reduction="none")(r_seq.view(-1, int(vocab_size)), seq.view(-1))
            err_seq = ce_loss.view(num.size(0), -1).mean(dim=1)

            z_vectors.append(z.detach().cpu().numpy())
            raw_num_list.append(num.cpu())
            raw_ctx_dyn_list.append(ctx_dyn.cpu())
            raw_ctx_stat_list.append(ctx_stat.cpu())
            raw_seq_list.append(seq.cpu())
            err_num_all.append(err_num.detach().cpu().numpy())
            err_ctx_all.append(err_ctx.detach().cpu().numpy())
            err_seq_all.append(err_seq.detach().cpu().numpy())
            all_labels.extend(batch["label"].numpy())
            all_user_dates.extend(batch["user_date"])
            all_indices.append(np.arange(global_idx, global_idx + int(z.size(0)), dtype=int))
            global_idx += int(z.size(0))

    z_vectors = np.concatenate(z_vectors, axis=0)
    raw_num = torch.cat(raw_num_list, dim=0)
    raw_ctx_dyn = torch.cat(raw_ctx_dyn_list, dim=0)
    raw_ctx_stat = torch.cat(raw_ctx_stat_list, dim=0)
    raw_seq = torch.cat(raw_seq_list, dim=0)
    err_num_all = np.concatenate(err_num_all, axis=0); err_ctx_all = np.concatenate(err_ctx_all, axis=0); err_seq_all = np.concatenate(err_seq_all, axis=0)
    all_labels = np.array(all_labels); all_indices = np.concatenate(all_indices, axis=0)
    score_raw = w_num * err_num_all + w_ctx * err_ctx_all + w_seq * err_seq_all

    # 初始化原型相关变量，确保在关闭 TTA 时代码逻辑依然完整
    margin_dict = {d: np.zeros(len(all_labels)) for d in ("num", "dyn", "seq")}
    dist_dict = {d: np.zeros(len(all_labels)) for d in ("num", "dyn", "seq")}
    status_dict = {d: np.full(len(all_labels), -2, dtype=int) for d in ("num", "dyn", "seq")}
    prototype_test_summary = {d: {"old_k": 0, "new_k": 0, "n_drift": 0, "new_clusters": 0} for d in ("num", "dyn", "seq")}
    total_margin = np.zeros(len(all_labels))
    new_cluster_indices = []
    new_cluster_data = None

    cls_metrics = metrics_at_best_f1(all_labels, score_raw) or fallback_metrics_single_class(all_labels, score_raw)

    if enable_tta:
        z_vectors_tensor = torch.tensor(z_vectors, device=device)
        score_raw_tensor = torch.from_numpy(score_raw).to(device)
        z_dim = model.z_dim
        Z_dict = {
            "num": F.normalize(z_vectors_tensor[:, :z_dim], p=2, dim=-1),
            "dyn": F.normalize(z_vectors_tensor[:, z_dim:2*z_dim], p=2, dim=-1),
            "seq": F.normalize(z_vectors_tensor[:, 2*z_dim:], p=2, dim=-1)
        }

        if not hasattr(model, 'proto_radii') or model.proto_radii is None:
            model.proto_radii = {d: torch.ones(getattr(model, f"proto_layer_{d}").prototypes.size(0), device=device) * 0.4 for d in ("num", "dyn", "seq")}

        # 1. 预计算每个维度的距离、命中原型和判定结果
        dim_info = {}
        for dim_key, layer_name in [("num", "proto_layer_num"), ("dyn", "proto_layer_dyn"), ("seq", "proto_layer_seq")]:
            layer = getattr(model, layer_name)
            prototypes = layer.prototypes.data
            radii = model.proto_radii[dim_key].to(device)
            Z = Z_dict[dim_key]
            
            dists = torch.cdist(Z, F.normalize(prototypes, p=2, dim=-1), p=2)
            min_dist, hit_id = torch.min(dists, dim=1)
            inner_thresholds = radii[hit_id] * tta_radius_tolerance
            
            is_inlier = min_dist <= inner_thresholds
            # 记录该维度的漂移状态 (用于后续一票否决)
            is_drift_dim = min_dist > inner_thresholds
            
            # EMA 更新现有原型 (保持原逻辑：只用顺应的样本更新)
            for k in range(prototypes.size(0)):
                k_mask = (hit_id == k) & is_inlier
                if k_mask.any():
                    prototypes[k].copy_((1 - tta_ema_alpha) * prototypes[k] + tta_ema_alpha * torch.mean(Z[k_mask], dim=0))
            
            dim_info[dim_key] = {
                "min_dist": min_dist,
                "hit_id": hit_id,
                "is_inlier": is_inlier,
                "is_drift": is_drift_dim,
                "prototypes": prototypes,
                "radii": radii,
                "layer": layer
            }

        # 2. 一票否决：只要有一个维度偏离，整体判定为 Drift
        is_drift_global = dim_info["num"]["is_drift"] | dim_info["dyn"]["is_drift"] | dim_info["seq"]["is_drift"]
        
        # 3. 重构误差防火墙：排除高重构误差的潜在威胁样本 (极其重要)
        baseline_thresh = float(cls_metrics["threshold"])
        is_clean_drift = is_drift_global & (score_raw_tensor < baseline_thresh * 1.5)
        
        drift_indices = torch.nonzero(is_clean_drift).view(-1)
        n_new_clusters = 0
        res_labels = np.full(drift_indices.numel(), -1, dtype=int)
        
        # 4. 联合空间特征拼接与聚类 (Joint Clustering)
        if drift_indices.numel() >= low_score_anomaly_dbscan_min_samples:
            # 拼接三个维度的 Z 特征 [N_drift, z_dim * 3]
            Z_concat_drift = torch.cat([
                Z_dict["num"][drift_indices],
                Z_dict["dyn"][drift_indices],
                Z_dict["seq"][drift_indices]
            ], dim=1).cpu().numpy()
            
            dbscan = DBSCAN(eps=low_score_anomaly_dbscan_eps, min_samples=low_score_anomaly_dbscan_min_samples, metric='euclidean')
            res_labels = dbscan.fit_predict(Z_concat_drift)
            
            unique_c = set(res_labels) - {-1}
            for c in unique_c:
                c_mask = (res_labels == c)
                curr_drift_idx = drift_indices[c_mask]
                
                # 记录新发现簇的样本索引，用于后续微调
                new_cluster_indices.extend(curr_drift_idx.cpu().numpy().tolist())
                
                # 同步更新三个维度的原型层
                for dim_key in ["num", "dyn", "seq"]:
                    Z_c = Z_dict[dim_key][curr_drift_idx]
                    centroid = F.normalize(torch.mean(Z_c, dim=0, keepdim=True), p=2, dim=-1)
                    # 计算该维度下的 95% 分位数作为新半径
                    r_95 = max(torch.quantile(torch.cdist(Z_c, centroid, p=2).view(-1), 0.95).item(), 0.1)
                    
                    layer = dim_info[dim_key]["layer"]
                    layer.prototypes = nn.Parameter(torch.cat([layer.prototypes.data, centroid], dim=0))
                    model.proto_radii[dim_key] = torch.cat([model.proto_radii[dim_key], torch.tensor([r_95], device=device)], dim=0)
                
                n_new_clusters += 1

        # 5. 重新计算最终距离和 Margin (使用更新后的原型)
        for dim_key in ["num", "dyn", "seq"]:
            layer = getattr(model, f"proto_layer_{dim_key}")
            final_protos = F.normalize(layer.prototypes.data, p=2, dim=-1)
            final_radii = model.proto_radii[dim_key].to(device)
            Z = Z_dict[dim_key]
            
            final_dists = torch.cdist(Z, final_protos, p=2)
            f_min_dist, f_hit_id = torch.min(final_dists, dim=1)
            
            margin = torch.clamp(f_min_dist - (final_radii[f_hit_id] * tta_radius_tolerance), min=0.0)
            
            # 状态记录：-2: 顺应, -1: 离异, >=0: 新簇
            status_labels = np.full(Z.shape[0], -1, dtype=int) # 默认为离异
            is_inlier_final = f_min_dist <= (final_radii[f_hit_id] * tta_radius_tolerance)
            status_labels[is_inlier_final.cpu().numpy()] = -2
            
            # 如果样本属于联合空间中的新簇，则标记对应的簇 ID
            if n_new_clusters > 0:
                drift_np_idx = drift_indices.cpu().numpy()
                valid_drift_mask = (res_labels >= 0)
                status_labels[drift_np_idx[valid_drift_mask]] = res_labels[valid_drift_mask]

            dist_dict[dim_key] = f_min_dist.cpu().numpy()
            margin_dict[dim_key] = margin.cpu().numpy()
            status_dict[dim_key] = status_labels
            prototype_test_summary[dim_key] = {
                "old_k": dim_info[dim_key]["prototypes"].size(0),
                "new_k": final_protos.size(0),
                "n_drift": drift_indices.numel(),
                "new_clusters": n_new_clusters
            }
            logger.info(f"   -> [{dim_key}] 空间分流 (联合判定): 顺应=%d | 偏离聚类池=%d | 演进新原型=%d", 
                        int(is_inlier_final.sum()), drift_indices.numel(), n_new_clusters)

        total_margin = margin_dict["num"] + margin_dict["dyn"] + margin_dict["seq"]
        
    # 最终异常得分计算
    anomaly_scores = score_raw + (proto_penalty_weight * total_margin)
    cls_metrics = metrics_at_best_f1(all_labels, anomaly_scores) or fallback_metrics_single_class(all_labels, anomaly_scores)

    # ---------------------------------------------------------
    # 终端日志：正样本细粒度行为安全画像监控
    # ---------------------------------------------------------
    if enable_pos_profiling:
        pos_mask = (all_labels == 1)
        if pos_mask.any():
            logger.info(f"   === 威胁(正样本)原型空间安全画像 ===")
            logger.info(f"   -> 真实威胁样本总判定离异率 (超判定线比例): {np.mean(total_margin[pos_mask] > 0):.2%}")
            for d in ["num", "dyn", "seq"]:
                p_status = status_dict[d][pos_mask]
                logger.info(f"   -> [{d}] 均距={np.mean(dist_dict[d][pos_mask]):.4f} | 伪装在旧模式内={np.mean(p_status == -2):.1%} | 突变被判定为确定游离异常={np.mean(p_status == -1):.1%} | 误入良性新模式簇={np.mean(p_status >= 0):.1%}")

    # ---------------------------------------------------------
    # 各项关键指标评估与落表保存
    # ---------------------------------------------------------
    auc, eer, pr_auc, p_at_100 = None, None, None, None
    dr_at_budgets, p_at_k, fpr_at_recall = {}, {}, {}
    pr_curve_path, score_dist_path, low_score_table_path = None, None, None

    if int(np.sum(all_labels)) > 0 and int(np.sum(all_labels == 0)) > 0:
        auc = roc_auc_score(all_labels, anomaly_scores)
        eer = compute_eer(all_labels, anomaly_scores)
        dr_at_budgets = detection_rate_at_budgets(all_labels, anomaly_scores, budgets=budget_list)
        pr_auc = average_precision_score(all_labels, anomaly_scores)
        p_at_100 = precision_at_k(all_labels, anomaly_scores, k=100)
        p_at_k = precision_at_k_multi(all_labels, anomaly_scores, ks=topk_list)
        fpr_at_recall = fpr_at_target_recalls(all_labels, anomaly_scores, target_recalls=target_recalls)
        
        if figure_dir is None: figure_dir = os.path.join(os.path.dirname(test_pkl), "evaluation_figures")
        os.makedirs(figure_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(test_pkl))[0]
        pr_curve_path, score_dist_path = os.path.join(figure_dir, f"{base}_pr_curve.png"), os.path.join(figure_dir, f"{base}_score_distribution.png")
        _save_pr_curve(all_labels, anomaly_scores, pr_curve_path)
        _save_score_distribution(all_labels, anomaly_scores, score_dist_path)
        try: _save_latent_space_tsne(z_vectors, all_labels, model, os.path.join(figure_dir, f"{base}_tsne_latent.png"))
        except Exception as e: logger.error("t-SNE 失败: %s", e)
        
        logger.info(
            "-> 性能指标: ROC-AUC=%.4f, EER=%.4f, PR-AUC=%.4f, P@100=%.4f, P@K=%s",
            auc,
            eer,
            pr_auc,
            p_at_100,
            {k: round(v, 4) for k, v in p_at_k.items()},
        )
    
    fpr = cls_metrics.get("fpr")
    tpr = cls_metrics.get("tpr")
    logger.info(
        "-> 单点评估(best F1 threshold=%.6f): TP=%d FP=%d TN=%d FN=%d | FPR=%s TPR=%s | Precision=%.4f Recall=%.4f F1=%.4f F2=%.4f",
        float(cls_metrics.get("threshold", 0.0)),
        int(cls_metrics.get("tp", 0)),
        int(cls_metrics.get("fp", 0)),
        int(cls_metrics.get("tn", 0)),
        int(cls_metrics.get("fn", 0)),
        "N/A" if fpr is None else "{:.6f}".format(float(fpr)),
        "N/A" if tpr is None else "{:.6f}".format(float(tpr)),
        float(cls_metrics.get("precision", 0.0)),
        float(cls_metrics.get("recall", 0.0)),
        float(cls_metrics.get("f1", 0.0)),
        float(cls_metrics.get("f2", 0.0)),
    )

    if fpr_at_recall:
        for target in sorted(fpr_at_recall.keys()):
            item = fpr_at_recall[target]
            if item is None:
                logger.info("-> FPR@Recall=%.2f: 不可达", float(target))
            else:
                logger.info(
                    "-> FPR@Recall=%.2f: FPR=%.6f | thr=%.6f | Precision=%.4f | TP=%d FP=%d TN=%d FN=%d",
                    float(target),
                    float(item["fpr"]),
                    float(item["threshold"]),
                    float(item["precision"]),
                    int(item["tp"]),
                    int(item["fp"]),
                    int(item["tn"]),
                    int(item["fn"]),
                )

    if dump_low_score_anomalies:
        base = os.path.splitext(os.path.basename(test_pkl))[0]
        table_dir = os.path.join(os.path.dirname(figure_dir if figure_dir else test_pkl), "evaluation_tables")
        os.makedirs(table_dir, exist_ok=True)
        low_score_table_path = os.path.join(table_dir, f"{base}_anomaly_analysis.csv")
        
        pos_idx = np.where(all_labels == 1)[0]
        pos_idx_sorted = pos_idx[np.argsort(-anomaly_scores[pos_idx])]
        neg_idx = np.where(all_labels == 0)[0]
        top_fp_idx = neg_idx[np.argsort(-anomaly_scores[neg_idx])[:low_score_anomaly_top_n]]
        chosen = np.concatenate([pos_idx_sorted, top_fp_idx])
        
        map_status = lambda s: "旧原型顺应" if s == -2 else ("极远端确定离异异常" if s == -1 else f"参与新模式簇_{s}")

        fieldnames = [
            "sample_index", "user_date", "label", "score_final", "score_recon_raw", "is_total_outlier",
            "dist_num", "dist_dyn", "dist_seq", "margin_num", "margin_dyn", "margin_seq",
            "status_num", "status_dyn", "status_seq"
        ]
        with open(low_score_table_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for idx in chosen:
                w.writerow({
                    "sample_index": int(all_indices[idx]), "user_date": str(all_user_dates[idx]), "label": int(all_labels[idx]),
                    "score_final": float(anomaly_scores[idx]), "score_recon_raw": float(score_raw[idx]),
                    "is_total_outlier": "Yes" if total_margin[idx] > 0 else "No",
                    "dist_num": float(dist_dict["num"][idx]), "dist_dyn": float(dist_dict["dyn"][idx]), "dist_seq": float(dist_dict["seq"][idx]),
                    "margin_num": float(margin_dict["num"][idx]), "margin_dyn": float(margin_dict["dyn"][idx]), "margin_seq": float(margin_dict["seq"][idx]),
                    "status_num": map_status(status_dict["num"][idx]), "status_dyn": map_status(status_dict["dyn"][idx]), "status_seq": map_status(status_dict["seq"][idx])
                })
        logger.info(">>> 全量威胁与对比高分样本多维指标落库成功: %s", low_score_table_path)

    if extract_new_clusters and len(new_cluster_indices) > 0:
        new_cluster_indices = np.unique(new_cluster_indices)
        new_cluster_data = {
            "num": raw_num[new_cluster_indices],
            "ctx_dynamic": raw_ctx_dyn[new_cluster_indices],
            "ctx_static": raw_ctx_stat[new_cluster_indices],
            "seq": raw_seq[new_cluster_indices]
        }
    else:
        new_cluster_data = None

    return {
        "z_vectors": z_vectors, "scores": anomaly_scores, "scores_raw": score_raw, 
        "err_num": err_num_all, "err_ctx": err_ctx_all, "err_seq": err_seq_all,
        "labels": all_labels, "user_dates": all_user_dates, "indices": all_indices, 
        "auc": auc, "eer": eer, "dr_at_budgets": dr_at_budgets,
        "pr_auc": pr_auc, "p_at_100": p_at_100, "p_at_k": p_at_k, "fpr_at_recall": fpr_at_recall, "metrics": cls_metrics,
        "prototype_test_summary": prototype_test_summary, "new_cluster_data": new_cluster_data,
    }
