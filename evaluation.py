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
    roc_curve,  # 新增：用于快速计算 EER
)

from training import CERTDataset

logger = logging.getLogger(__name__)


def precision_at_k(y_true, scores, k: int = 100) -> float:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    k = min(int(k), len(y_sorted))
    if k == 0:
        return 0.0
    top_k_labels = y_sorted[:k]
    return float(np.sum(top_k_labels == 1) / k)


def precision_at_k_multi(y_true, scores, ks=(50, 100)) -> dict[int, float]:
    out: dict[int, float] = {}
    for k in ks:
        out[int(k)] = precision_at_k(y_true, scores, k=int(k))
    return out


def compute_eer(y_true, scores):
    """
    计算等错误率 (EER, Equal Error Rate)，即 FPR 约等于 FNR (1 - TPR) 时的错误率。
    衡量模型在误报和漏报之间的平衡性。
    """
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    # 寻找 FPR 和 FNR 差距最小的点
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer)


def detection_rate_at_budgets(y_true, scores, budgets=(0.05, 0.10, 0.15)):
    """
    计算特定调查预算（前 k% 高分样本）下的检测率 (DR / TPR / Recall)。
    对应 LAN 论文中的 DR@5%, DR@10%, DR@15%。
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    total_pos = np.sum(y_true == 1)
    if total_pos == 0:
        return {float(b): 0.0 for b in budgets}
        
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    
    out = {}
    total_samples = len(y_true)
    for b in budgets:
        # 取前 budget 比例的样本量
        k = int(np.ceil(b * total_samples))
        k = min(k, total_samples)
        if k == 0:
            out[float(b)] = 0.0
        else:
            top_k_labels = y_sorted[:k]
            tp = np.sum(top_k_labels == 1)
            # DR (Detection Rate) = TP / Total Positives
            out[float(b)] = float(tp / total_pos)
    return out


def _binary_curve_from_scores(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    total_pos = int(np.sum(y_true == 1))
    total_neg = int(np.sum(y_true == 0))
    if total_pos == 0 or total_neg == 0:
        return None
    order = np.argsort(-scores)
    scores_sorted = scores[order]
    y_sorted = y_true[order]
    is_pos = (y_sorted == 1).astype(int)
    is_neg = (y_sorted == 0).astype(int)
    cum_tp = np.cumsum(is_pos)
    cum_fp = np.cumsum(is_neg)
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
    f2 = np.divide((1 + 2 * 2) * precision * recall, (2 * 2) * precision + recall, out=np.zeros_like(precision, dtype=float), where=((2 * 2) * precision + recall) != 0)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "thresholds": thresholds,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "f1": f1,
        "f2": f2,
        "total_pos": total_pos,
        "total_neg": total_neg,
    }


def metrics_at_best_f1(y_true, scores):
    curve = _binary_curve_from_scores(y_true, scores)
    if curve is None:
        return None
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    best_i = int(np.argmax(curve["f1"]))
    best_thr = float(curve["thresholds"][best_i])
    y_pred = (scores >= best_thr).astype(int)
    tn2, fp2, fn2, tp2 = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp2 / (fp2 + tn2)) if (fp2 + tn2) > 0 else None
    tpr = float(tp2 / (tp2 + fn2)) if (tp2 + fn2) > 0 else None
    return {
        "threshold": best_thr,
        "tp": int(tp2),
        "fp": int(fp2),
        "tn": int(tn2),
        "fn": int(fn2),
        "fpr": fpr,
        "tpr": tpr,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def fpr_at_target_recalls(y_true, scores, target_recalls=(0.8, 0.9, 0.95)):
    curve = _binary_curve_from_scores(y_true, scores)
    if curve is None:
        return {float(r): None for r in target_recalls}
    out = {}
    recalls = curve["recall"]
    fprs = curve["fpr"]
    thrs = curve["thresholds"]
    for target in target_recalls:
        idx = np.where(recalls >= float(target))[0]
        if len(idx) == 0:
            out[float(target)] = None
            continue
        candidate_fprs = fprs[idx]
        min_rel_i = int(np.argmin(candidate_fprs))
        best_i = int(idx[min_rel_i])
        out[float(target)] = {
            "fpr": float(fprs[best_i]),
            "threshold": float(thrs[best_i]),
            "precision": float(curve["precision"][best_i]),
            "recall": float(recalls[best_i]),
            "tp": int(curve["tp"][best_i]),
            "fp": int(curve["fp"][best_i]),
            "tn": int(curve["tn"][best_i]),
            "fn": int(curve["fn"][best_i]),
        }
    return out


def _save_pr_curve(y_true, scores, save_path: str) -> bool:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    if int(np.sum(y_true == 1)) == 0 or int(np.sum(y_true == 0)) == 0:
        return False
    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR-AUC={pr_auc:.4f}", linewidth=2)
    plt.axvspan(0.7, 0.9, alpha=0.1, color="orange", label="Recall 0.7-0.9")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return True


def _save_score_distribution(y_true, scores, save_path: str) -> bool:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]
    if len(normal_scores) == 0 or len(anomaly_scores) == 0:
        return False
    bins = 60
    plt.figure(figsize=(8, 6))
    plt.hist(normal_scores, bins=bins, alpha=0.6, density=True, label=f"Normal (n={len(normal_scores)})")
    plt.hist(anomaly_scores, bins=bins, alpha=0.6, density=True, label=f"Anomaly (n={len(anomaly_scores)})")
    plt.xlabel("Reconstruction Loss")
    plt.ylabel("Density")
    plt.title("Score Distribution Overlap")
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return True


def fallback_metrics_single_class(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    total_pos = int(np.sum(y_true == 1))
    total_neg = int(np.sum(y_true == 0))
    if total_pos == 0 and total_neg == 0:
        raise ValueError("Empty y_true is not supported.")
    if total_pos == 0:
        thr = float("inf")
        y_pred = np.zeros_like(y_true, dtype=int)
    else:
        thr = float("-inf")
        y_pred = np.ones_like(y_true, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else None
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else None
    return {
        "threshold": thr,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "fpr": fpr,
        "tpr": tpr,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _get_multihead_prototypes_concat(model) -> np.ndarray | None:
    if not all(hasattr(model, name) for name in ("proto_layer_num", "proto_layer_dyn", "proto_layer_seq")):
        return None
    proto_num = getattr(model.proto_layer_num, "prototypes", None)
    proto_dyn = getattr(model.proto_layer_dyn, "prototypes", None)
    proto_seq = getattr(model.proto_layer_seq, "prototypes", None)
    if not (torch.is_tensor(proto_num) and torch.is_tensor(proto_dyn) and torch.is_tensor(proto_seq)):
        return None
        
    p_num = F.normalize(proto_num, p=2, dim=1)
    p_dyn = F.normalize(proto_dyn, p=2, dim=1)
    p_seq = F.normalize(proto_seq, p=2, dim=1)
    
    # [修改] 由于各个维度被独立剪枝，K值可能不同。使用 Zero-Padding 对齐到最大的 k
    max_k = max(p_num.size(0), p_dyn.size(0), p_seq.size(0))
    
    def pad_proto(p):
        if p.size(0) == max_k: return p
        pad_size = max_k - p.size(0)
        return F.pad(p, (0, 0, 0, pad_size), "constant", 0.0)
        
    prototypes = torch.cat([pad_proto(p_num), pad_proto(p_dyn), pad_proto(p_seq)], dim=1)
    return prototypes.detach().cpu().numpy()


def _save_latent_space_tsne(z_vectors: np.ndarray, labels: np.ndarray, model, save_path: str, max_samples: int = 5000) -> bool:
    z_vectors = np.asarray(z_vectors)
    labels = np.asarray(labels).astype(int)
    prototypes = _get_multihead_prototypes_concat(model)
    if prototypes is None:
        return False
    num_prototypes = int(prototypes.shape[0])
    if int(z_vectors.shape[0]) > int(max_samples):
        np.random.seed(42)
        indices = np.random.choice(int(z_vectors.shape[0]), int(max_samples), replace=False)
        z_sample = z_vectors[indices]
        label_sample = labels[indices]
    else:
        z_sample = z_vectors
        label_sample = labels
    all_features = np.vstack([z_sample, prototypes])
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
    features_2d = tsne.fit_transform(all_features)
    z_2d = features_2d[:-num_prototypes]
    p_2d = features_2d[-num_prototypes:]
    plt.figure(figsize=(12, 10))
    normal_mask = label_sample == 0
    if np.any(normal_mask):
        plt.scatter(z_2d[normal_mask, 0], z_2d[normal_mask, 1], alpha=0.3, s=15, c="#1f77b4", label=f"Normal (n={np.sum(normal_mask)})")
    anomaly_mask = label_sample == 1
    if np.any(anomaly_mask):
        plt.scatter(z_2d[anomaly_mask, 0], z_2d[anomaly_mask, 1], alpha=0.6, s=25, c="#ff7f0e", marker="x", label=f"Anomaly (n={np.sum(anomaly_mask)})")
    plt.scatter(p_2d[:, 0], p_2d[:, 1], marker="*", c="#d62728", s=400, edgecolor="black", linewidths=1.5, label=f"Prototypes (k={num_prototypes})")
    for i in range(num_prototypes):
        plt.annotate(str(i), (p_2d[i, 0], p_2d[i, 1]), xytext=(5, 5), textcoords="offset points", fontsize=12, fontweight="bold")
    plt.title(f"t-SNE Latent Space & Prototypes (k={num_prototypes})", fontsize=16)
    plt.legend(loc="best", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return True


def _extract_user_from_user_date(user_date) -> str:
    s = str(user_date)
    if "_" in s:
        return s.rsplit("_", 1)[0]
    return s


def evaluate_and_extract_z(
    model,
    test_pkl: str,
    batch_size: int = 128,
    device: str | torch.device = "cuda",
    weights: dict | None = None,
    vocab_size: int = 8,
    topk_list=(50, 100),
    target_recalls=(0.8, 0.9, 0.95),
    budget_list=(0.05, 0.10, 0.15),
    figure_dir: str | None = None,
    dump_low_score_anomalies: bool = True,
    low_score_anomaly_top_ratio: float = 0.5,
    low_score_anomaly_top_n: int = 200,
    low_score_anomaly_dbscan_eps: float = 0.5,
    low_score_anomaly_dbscan_min_samples: int = 10,
):
    logger.info("[%s] >>> 开始在测试集上评估并提取 Z: %s", datetime.now(), os.path.basename(test_pkl))
    test_dataset = CERTDataset(test_pkl)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=int(batch_size), shuffle=False)
    if weights is None:
        weights = {"num": 1.0, "ctx": 1.0, "seq": 0.1}
    weights = {"num": float(weights.get("num", 1.0)), "ctx": float(weights.get("ctx", 1.0)), "seq": float(weights.get("seq", 0.1))}
    model.eval()

    z_vectors = []
    anomaly_scores = []
    anomaly_scores_raw = []
    err_num_all = []
    err_ctx_all = []
    err_seq_all = []
    all_labels = []
    all_user_dates = []
    all_indices = []
    global_idx = 0
    proto_min_dist_all = None
    proto_min_dist_by_dim_np = None
    proto_hit_id_by_dim_np = None
    proto_hit_id_in_radius_by_dim_np = None
    proto_min_dist_in_radius_by_dim_np = None
    prototype_test_summary = None

    with torch.no_grad():
        for batch in test_loader:
            num = batch["num"].to(device)
            seq = batch["seq"].to(device)
            ctx_dynamic = batch.get("ctx_dynamic", None)
            ctx_static = batch.get("ctx_static", None)
            if ctx_dynamic is None or ctx_static is None:
                ctx = batch["ctx"].to(device)
                raise ValueError(f"static_fused_three_z requires ctx_dynamic/ctx_static but got ctx shape {tuple(ctx.shape)}")
            ctx_dynamic = ctx_dynamic.to(device)
            ctx_static = ctx_static.to(device)
            ctx = torch.cat([ctx_dynamic, ctx_static], dim=1)

            outputs = model(num, ctx_dynamic, ctx_static, seq)
            z = outputs[0]
            r_num = outputs[1]
            r_ctx = outputs[2]
            r_seq = outputs[3]

            err_num = torch.mean((r_num - num) ** 2, dim=1)
            err_ctx = torch.zeros(num.size(0), device=num.device)
            if torch.is_tensor(r_ctx) and r_ctx.shape == ctx.shape:
                err_ctx = torch.mean((r_ctx - ctx) ** 2, dim=1)
            r_seq_flat = r_seq.view(-1, int(vocab_size))
            seq_flat = seq.view(-1)
            ce_unreduced = nn.CrossEntropyLoss(ignore_index=0, reduction="none")(r_seq_flat, seq_flat)
            err_seq = ce_unreduced.view(num.size(0), -1).mean(dim=1)

            score_raw = weights["num"] * err_num + weights["ctx"] * err_ctx + weights["seq"] * err_seq

            batch_size_curr = int(z.size(0))
            batch_indices = np.arange(global_idx, global_idx + batch_size_curr, dtype=int)
            global_idx += batch_size_curr

            z_vectors.append(z.detach().cpu().numpy())
            anomaly_scores.append(score_raw.detach().cpu().numpy())
            anomaly_scores_raw.append(score_raw.detach().cpu().numpy())
            err_num_all.append(err_num.detach().cpu().numpy())
            err_ctx_all.append(err_ctx.detach().cpu().numpy())
            err_seq_all.append(err_seq.detach().cpu().numpy())
            all_labels.extend(batch["label"].numpy())
            all_user_dates.extend(batch["user_date"])
            all_indices.append(batch_indices)

    z_vectors = np.concatenate(z_vectors, axis=0)
    anomaly_scores = np.concatenate(anomaly_scores, axis=0)
    anomaly_scores_raw = np.concatenate(anomaly_scores_raw, axis=0)
    err_num_all = np.concatenate(err_num_all, axis=0)
    err_ctx_all = np.concatenate(err_ctx_all, axis=0)
    err_seq_all = np.concatenate(err_seq_all, axis=0)
    all_labels = np.array(all_labels)
    all_indices = np.concatenate(all_indices, axis=0)

    auc = None
    eer = None                # 新增
    dr_at_budgets = {}
    pr_auc = None
    p_at_100 = None
    p_at_k = {}
    fpr_at_recall = {}
    pr_curve_path = None
    score_dist_path = None
    if int(np.sum(all_labels)) > 0 and int(np.sum(all_labels == 0)) > 0:
        auc = roc_auc_score(all_labels, anomaly_scores)
        eer = compute_eer(all_labels, anomaly_scores)
        dr_at_budgets = detection_rate_at_budgets(all_labels, anomaly_scores, budgets=budget_list)
        pr_auc = average_precision_score(all_labels, anomaly_scores)
        p_at_100 = precision_at_k(all_labels, anomaly_scores, k=100)
        p_at_k = precision_at_k_multi(all_labels, anomaly_scores, ks=topk_list)
        fpr_at_recall = fpr_at_target_recalls(all_labels, anomaly_scores, target_recalls=target_recalls)
        if figure_dir is None:
            figure_dir = os.path.join(os.path.dirname(test_pkl), "evaluation_figures")
        os.makedirs(figure_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(test_pkl))[0]
        pr_curve_path = os.path.join(figure_dir, f"{base}_pr_curve.png")
        score_dist_path = os.path.join(figure_dir, f"{base}_score_distribution.png")
        _save_pr_curve(all_labels, anomaly_scores, pr_curve_path)
        _save_score_distribution(all_labels, anomaly_scores, score_dist_path)
        tsne_path = os.path.join(figure_dir, f"{base}_tsne_latent.png")
        try:
            _save_latent_space_tsne(z_vectors, all_labels, model, tsne_path)
        except Exception as e:
            logger.error("t-SNE 可视化失败: %s", e)
        logger.info(
            "-> 阈值无关指标: ROC-AUC=%.4f, EER=%.4f, PR-AUC=%.4f, P@100=%.4f, P@K=%s",
            auc,
            eer,  # 记录 EER
            pr_auc,
            p_at_100,
            {k: round(v, 4) for k, v in p_at_k.items()},
        )
        logger.info("-> 可视化已保存: PR曲线=%s | 分布图=%s", pr_curve_path, score_dist_path)
    else:
        logger.info("-> 测试集中仅含单一类别，跳过 ROC-AUC/PR-AUC/P@K/FPR@Recall 和可视化。")

    cls_metrics = metrics_at_best_f1(all_labels, anomaly_scores)
    if cls_metrics is None:
        cls_metrics = fallback_metrics_single_class(all_labels, anomaly_scores)

    fpr = cls_metrics.get("fpr") if isinstance(cls_metrics, dict) else None
    tpr = cls_metrics.get("tpr") if isinstance(cls_metrics, dict) else None
    logger.info(
        "-> 单点评估(best F1 threshold=%.6f): TP=%d FP=%d TN=%d FN=%d | FPR=%s TPR=%s | Precision=%.4f Recall=%.4f F1=%.4f F2=%.4f",
        float(cls_metrics.get("threshold")),
        int(cls_metrics.get("tp")),
        int(cls_metrics.get("fp")),
        int(cls_metrics.get("tn")),
        int(cls_metrics.get("fn")),
        "N/A" if fpr is None else "{:.6f}".format(float(fpr)),
        "N/A" if tpr is None else "{:.6f}".format(float(tpr)),
        float(cls_metrics.get("precision")),
        float(cls_metrics.get("recall")),
        float(cls_metrics.get("f1")),
        float(cls_metrics.get("f2")),
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

    low_score_table_path = None
    if dump_low_score_anomalies:
        anomaly_idx = np.where(all_labels == 1)[0]
        if len(anomaly_idx) > 0:
            base = os.path.splitext(os.path.basename(test_pkl))[0]
            if figure_dir is None:
                figure_dir = os.path.join(os.path.dirname(test_pkl), "evaluation_figures")
            run_dir = os.path.dirname(figure_dir)
            table_dir = os.path.join(run_dir, "evaluation_tables")
            os.makedirs(table_dir, exist_ok=True)
            low_score_table_path = os.path.join(table_dir, f"{base}_low_score_anomalies.csv")
            scores_anom = anomaly_scores[anomaly_idx]
            order = np.argsort(scores_anom)
            n_by_ratio = int(np.ceil(float(low_score_anomaly_top_ratio) * len(anomaly_idx)))
            n_dump = int(min(len(anomaly_idx), max(1, min(int(low_score_anomaly_top_n), n_by_ratio))))
            chosen = anomaly_idx[order[:n_dump]]
            scores_all_final = np.asarray(anomaly_scores).astype(float)
            scores_all_raw = np.asarray(anomaly_scores_raw).astype(float)
            order_all_final = np.argsort(scores_all_final)
            ranks_all_final = np.empty_like(order_all_final, dtype=int)
            ranks_all_final[order_all_final] = np.arange(1, len(order_all_final) + 1, dtype=int)
            order_all_raw = np.argsort(scores_all_raw)
            ranks_all_raw = np.empty_like(order_all_raw, dtype=int)
            ranks_all_raw[order_all_raw] = np.arange(1, len(order_all_raw) + 1, dtype=int)
            denom_all = max(1, (len(scores_all_final) - 1))
            thr = float(cls_metrics["threshold"]) if cls_metrics and cls_metrics.get("threshold") is not None else None
            w_num = float(weights.get("num", 1.0)) if isinstance(weights, dict) else 1.0
            w_ctx = float(weights.get("ctx", 1.0)) if isinstance(weights, dict) else 1.0
            w_seq = float(weights.get("seq", 0.1)) if isinstance(weights, dict) else 0.1
            fieldnames = [
                "sample_index",
                "user_date",
                "label",
                "score",
                "threshold_best_f1",
                "pred_anomaly_best_f1",
                "margin_to_threshold",
                "err_num",
                "err_ctx",
                "err_seq",
                "contrib_num",
                "contrib_ctx",
                "contrib_seq",
                "contrib_num_frac",
                "contrib_ctx_frac",
                "contrib_seq_frac",
                "anomaly_score_rank_asc",
                "anomaly_score_percentile",
                "score_raw_rank_asc",
                "score_raw_percentile",
            ]
            with open(low_score_table_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for idx in chosen:
                    rank_final_all = int(ranks_all_final[idx])
                    percentile_final_all = float((rank_final_all - 1) / denom_all)
                    rank_raw_all = int(ranks_all_raw[idx])
                    percentile_raw_all = float((rank_raw_all - 1) / denom_all)
                    score_val = float(anomaly_scores[idx])
                    score_raw = float(anomaly_scores_raw[idx])
                    contrib_num = float(w_num * err_num_all[idx])
                    contrib_ctx = float(w_ctx * err_ctx_all[idx])
                    contrib_seq = float(w_seq * err_seq_all[idx])
                    frac_num = ""
                    frac_ctx = ""
                    frac_seq = ""
                    if np.isfinite(score_raw) and abs(score_raw) > 1e-12:
                        frac_num = float(contrib_num / score_raw)
                        frac_ctx = float(contrib_ctx / score_raw)
                        frac_seq = float(contrib_seq / score_raw)
                    pred_best_f1 = ""
                    margin_to_thr = ""
                    if thr is not None:
                        pred_best_f1 = bool(score_val >= thr)
                        margin_to_thr = float(score_val - thr)
                    w.writerow(
                        {
                            "sample_index": int(all_indices[idx]),
                            "user_date": str(all_user_dates[idx]),
                            "label": int(all_labels[idx]),
                            "score": score_val,
                            "threshold_best_f1": "" if thr is None else float(thr),
                            "pred_anomaly_best_f1": pred_best_f1,
                            "margin_to_threshold": margin_to_thr,
                            "err_num": float(err_num_all[idx]),
                            "err_ctx": float(err_ctx_all[idx]),
                            "err_seq": float(err_seq_all[idx]),
                            "contrib_num": contrib_num,
                            "contrib_ctx": contrib_ctx,
                            "contrib_seq": contrib_seq,
                            "contrib_num_frac": frac_num,
                            "contrib_ctx_frac": frac_ctx,
                            "contrib_seq_frac": frac_seq,
                            "anomaly_score_rank_asc": rank_final_all,
                            "anomaly_score_percentile": percentile_final_all,
                            "score_raw_rank_asc": rank_raw_all,
                            "score_raw_percentile": percentile_raw_all,
                        }
                    )
            logger.info(">>> 已落表低分异常样本: %s | anomalies=%d | dumped=%d", low_score_table_path, int(len(anomaly_idx)), int(len(chosen)))

    return {
        "z_vectors": z_vectors,
        "scores": anomaly_scores,
        "scores_raw": anomaly_scores_raw,
        "err_num": err_num_all,
        "err_ctx": err_ctx_all,
        "err_seq": err_seq_all,
        "labels": all_labels,
        "user_dates": all_user_dates,
        "indices": all_indices,
        "auc": auc,
        "eer": eer,                                  # 新增返回值
        "dr_at_budgets": dr_at_budgets,              # 新增返回值
        "pr_auc": pr_auc,
        "p_at_100": p_at_100,
        "p_at_k": p_at_k,
        "fpr_at_recall": fpr_at_recall,
        "metrics": cls_metrics,
        "pr_curve_path": pr_curve_path,
        "score_dist_path": score_dist_path,
        "low_score_table_path": low_score_table_path,
        "prototype_min_dist": proto_min_dist_all,
        "prototype_min_dist_by_dim": proto_min_dist_by_dim_np,
        "prototype_hit_id_by_dim": proto_hit_id_by_dim_np,
        "prototype_hit_id_in_radius_by_dim": proto_hit_id_in_radius_by_dim_np,
        "prototype_min_dist_in_radius_by_dim": proto_min_dist_in_radius_by_dim_np,
        "prototype_test_summary": prototype_test_summary,
        "low_score_anomaly_dbscan_eps": float(low_score_anomaly_dbscan_eps),
        "low_score_anomaly_dbscan_min_samples": int(low_score_anomaly_dbscan_min_samples),
    }
