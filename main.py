import argparse
import csv
import logging
import os
import pickle
import re
from datetime import datetime

import numpy as np
import torch

from evaluation import evaluate_and_extract_z
from training import find_latest_checkpoint, load_model_checkpoint, save_model_checkpoint, train_model
from training import build_initial_memory_buffer, finetune_model_with_buffer

logger = logging.getLogger(__name__)


def setup_logging(log_path: str):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)


def _safe_path_segment(text, max_len: int = 80) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""
    s = s.replace(os.sep, "-")
    if os.altsep:
        s = s.replace(os.altsep, "-")
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-._")
    if len(s) > int(max_len):
        s = s[: int(max_len)].rstrip("-._")
    return s


def create_run_dir(base_output_dir: str, model_info: str | None = None, comment: str | None = None) -> str:
    os.makedirs(base_output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    parts = [
        _safe_path_segment(model_info, max_len=120),
        _safe_path_segment(comment, max_len=80),
        ts,
    ]
    run_name = "_".join([p for p in parts if p])
    run_dir = os.path.join(base_output_dir, run_name)
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def get_ordered_test_pkls(data_dir: str) -> list[str]:
    filenames = []
    for name in os.listdir(data_dir):
        if name == "test_historical.pkl" or re.fullmatch(r"test_month_\d{4}-\d{2}\.pkl", name):
            filenames.append(name)
    historical = []
    month = []
    other = []
    for name in filenames:
        if name == "test_historical.pkl":
            historical.append(name)
        elif name.startswith("test_month_"):
            month.append(name)
        else:
            other.append(name)
    month.sort()
    other.sort()
    ordered = historical + month + other
    return [os.path.join(data_dir, name) for name in ordered]


def append_result_row(csv_path: str, row: dict, fieldnames: list[str]):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="FD-ITDD: static_fused_three_z 训练与评估")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "both"], default="both")
    parser.add_argument("--data_dir", type=str, default="/workspace/wangyixin/datasets/CERT/FD-ITDD")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="/workspace/wangyixin/models/FD")
    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--seperate", type=int, choices=[0, 1, 2, 3, 4], default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda:3" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--comment", type=str, default="默认运行")
    parser.add_argument("--skip_train_if_exists", action="store_true")
    parser.add_argument("--disable_low_score_anomaly_dump", action="store_true")
    parser.add_argument("--low_score_anomaly_top_ratio", type=float, default=0.5)
    parser.add_argument("--low_score_anomaly_top_n", type=int, default=200)
    parser.add_argument("--low_score_anomaly_dbscan_eps", type=float, default=0.03)
    parser.add_argument("--low_score_anomaly_dbscan_min_samples", type=int, default=500)
    parser.add_argument("--num_prototypes", type=int, default=30)
    parser.add_argument("--prune_ratio", type=float, default=0.01)
    parser.add_argument("--disable_tta", action="store_true", help="关闭原型漂移适应机制 (TTA)")
    parser.add_argument("--tta_ema_alpha", type=float, default=0.05, help="TTA 动量更新系数")
    parser.add_argument("--tta_radius_tolerance", type=float, default=0.3, help="TTA 判定正常模式的半径容忍度")
    parser.add_argument("--proto_penalty_weight", type=float, default=1.0, help="原型离群惩罚权重")
    args = parser.parse_args()

    run_dir = create_run_dir(args.output_dir, model_info="static_fused_three_z", comment=args.comment)
    setup_logging(os.path.join(run_dir, "run.log"))
    logger.info(">>> FD-ITDD 项目启动")
    logger.info(">>> 输出目录: %s", run_dir)
    logger.info(">>> 使用计算设备: %s", args.device)

    device = torch.device(args.device)
    trained_model = None
    meta = None

    if args.mode in ["train", "both"]:
        ckpt_path = None
        if args.skip_train_if_exists:
            ckpt_path = find_latest_checkpoint(args.model_path, args.output_dir)
            if ckpt_path:
                logger.info(">>> 发现现有权重，跳过训练: %s", ckpt_path)
                trained_model, meta = load_model_checkpoint(ckpt_path, device=device)
        if trained_model is None:
            train_pkl_path = os.path.join(args.data_dir, "train_set.pkl")
            trained_model, meta = train_model(
                train_pkl=train_pkl_path,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                embed_dim=args.embed_dim,
                z_dim=args.z_dim,
                seperate=args.seperate,
                num_prototypes=args.num_prototypes,
                prune_ratio=args.prune_ratio, # 传递新增参数
                run_dir=run_dir,
            )
            ckpt_path = save_model_checkpoint(trained_model, meta, run_dir)
            logger.info(">>> 模型训练完成并保存: %s", ckpt_path)

    if args.mode in ["eval", "both"]:
        if trained_model is None:
            ckpt_path = find_latest_checkpoint(args.model_path, args.output_dir)
            if ckpt_path:
                logger.info(">>> 加载最新权重进行评估: %s", ckpt_path)
                trained_model, meta = load_model_checkpoint(ckpt_path, device=device)
            else:
                raise ValueError("未找到模型权重，无法进行评估")

        logger.info(">>> 开始评估与 TTA 持续适应阶段...")
        test_pkls = get_ordered_test_pkls(args.data_dir)
        if not test_pkls:
            raise ValueError(f"在 {args.data_dir} 中未找到测试集文件")

        # [Step 0]: 构建初始化历史经验库 Buffer
        train_pkl_path = os.path.join(args.data_dir, "train_set.pkl")
        memory_buffer = build_initial_memory_buffer(
            trained_model, train_pkl_path, device=device, samples_per_proto=50
        )

        results_csv_path_base = os.path.join(args.output_dir, "evaluation_csv")
        eval_figure_dir = os.path.join(run_dir, "evaluation_figures")
        os.makedirs(eval_figure_dir, exist_ok=True)
        summary = []

        for test_pkl in test_pkls:
            # [Step 1-3]: 推理与监控，发现新模式，执行 DBSCAN 与重打分
            result = evaluate_and_extract_z(
                model=trained_model,
                test_pkl=test_pkl,
                batch_size=args.batch_size,
                device=device,
                weights=meta["weights"],
                vocab_size=meta["vocab_size"],
                figure_dir=eval_figure_dir,
                dump_low_score_anomalies=not args.disable_low_score_anomaly_dump,
                low_score_anomaly_top_ratio=args.low_score_anomaly_top_ratio,
                low_score_anomaly_top_n=args.low_score_anomaly_top_n,
                low_score_anomaly_dbscan_eps=args.low_score_anomaly_dbscan_eps,
                low_score_anomaly_dbscan_min_samples=args.low_score_anomaly_dbscan_min_samples,
                enable_tta=not args.disable_tta,
                tta_ema_alpha=args.tta_ema_alpha,
                tta_radius_tolerance=args.tta_radius_tolerance,
                proto_penalty_weight=args.proto_penalty_weight,
            )

            # ... (原有的 CSV 打点和评估落表逻辑保持不变)
            base = os.path.splitext(os.path.basename(test_pkl))[0]
            out_pkl = os.path.join(run_dir, f"{base}_z_features.pkl")
            with open(out_pkl, "wb") as f:
                pickle.dump(result, f)

            n_samples = len(result["labels"])
            n_anomaly = int(np.sum(result["labels"]))
            summary.append(
                {
                    "test_pkl": os.path.basename(test_pkl),
                    "n_samples": n_samples,
                    "n_anomaly": n_anomaly,
                    "auc": result["auc"],
                    "metrics": result.get("metrics"),
                    "output_pkl": out_pkl,
                }
            )

            metrics = result.get("metrics") or {}
            csv_path = os.path.join(results_csv_path_base, f"{base}.csv")
            target_recalls = (0.8, 0.9, 0.95)
            extra_cols = []
            for r in target_recalls:
                r_str = "{:.2f}".format(float(r))
                extra_cols.extend([f"FPR@Recall={r_str}", f"Precision@Recall={r_str}", f"Thr@Recall={r_str}"])
            fieldnames = ["comment", "model", "acc", "AUC", "PR_AUC", "P@100", "Precision", "Recall", "F1", "F2", "TP", "FP", "TN", "FN"] + extra_cols
            row = {
                "comment": args.comment,
                "model": "static_fused_three_z",
                "acc": "{:.6f}".format(float(metrics["accuracy"])) if metrics.get("accuracy") is not None else "",
                "AUC": "{:.6f}".format(float(result["auc"])) if result.get("auc") is not None else "",
                "PR_AUC": "{:.6f}".format(float(result["pr_auc"])) if result.get("pr_auc") is not None else "",
                "P@100": "{:.6f}".format(float(result["p_at_100"])) if result.get("p_at_100") is not None else "",
                "Precision": "{:.6f}".format(float(metrics["precision"])) if metrics.get("precision") is not None else "",
                "Recall": "{:.6f}".format(float(metrics["recall"])) if metrics.get("recall") is not None else "",
                "F1": "{:.6f}".format(float(metrics["f1"])) if metrics.get("f1") is not None else "",
                "F2": "{:.6f}".format(float(metrics["f2"])) if metrics.get("f2") is not None else "",
                "TP": int(metrics["tp"]) if metrics.get("tp") is not None else "",
                "FP": int(metrics["fp"]) if metrics.get("fp") is not None else "",
                "TN": int(metrics["tn"]) if metrics.get("tn") is not None else "",
                "FN": int(metrics["fn"]) if metrics.get("fn") is not None else "",
            }
            fpr_at_recall = result.get("fpr_at_recall") or {}
            for r in target_recalls:
                r_f = float(r)
                r_str = "{:.2f}".format(r_f)
                item = fpr_at_recall.get(r_f)
                row[f"FPR@Recall={r_str}"] = "{:.6f}".format(float(item["fpr"])) if isinstance(item, dict) and item.get("fpr") is not None else ""
                row[f"Precision@Recall={r_str}"] = (
                    "{:.6f}".format(float(item["precision"])) if isinstance(item, dict) and item.get("precision") is not None else ""
                )
                row[f"Thr@Recall={r_str}"] = "{:.6f}".format(float(item["threshold"])) if isinstance(item, dict) and item.get("threshold") is not None else ""
            append_result_row(csv_path, row, fieldnames=fieldnames)

            auc_str = f"{result['auc']:.4f}" if result["auc"] is not None else "N/A"
            logger.info(">>> 测试完成: %s | samples=%d | anomalies=%d | auc=%s", base, n_samples, n_anomaly, auc_str)

            # [Step 4]: 提取 DBSCAN 发掘的新簇代表样本并触发微调
            new_cluster_data = result.get("new_cluster_data")
            if new_cluster_data and len(new_cluster_data["num"]) > 0:
                logger.info(">>> 发现新行为簇，正在使用 Memory Buffer 触发核心集经验回放微调...")
                trained_model, memory_buffer = finetune_model_with_buffer(
                    model=trained_model,
                    memory_buffer=memory_buffer,
                    new_cluster_data=new_cluster_data,
                    device=device,
                    vocab_size=meta["vocab_size"],
                    weights=meta["weights"],
                    seperate_method=args.seperate,
                    lr=1e-5,          # 极小学习率冻结底层语义
                    epochs=2,         # 极速 2 个 Epoch
                    anchor_weight=100.0 # 施加超强锚点损失
                )
            else:
                logger.info(">>> 本月数据无新模式固化，沿用当前 Memory Buffer 进入下一月度。")

        summary_path = os.path.join(run_dir, "evaluation_summary.pkl")
        with open(summary_path, "wb") as f:
            pickle.dump(summary, f)
        logger.info(">>> 全部测试完成，汇总已保存: %s", summary_path)


if __name__ == "__main__":
    main()

