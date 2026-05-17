import csv
import logging
import os
import pickle
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from models import StaticFusedThreeZAutoencoder

logger = logging.getLogger(__name__)


class CERTDataset(Dataset):
    def __init__(self, pkl_path: str):
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)
        if not isinstance(self.data, dict):
            raise ValueError(f"Invalid dataset pickle schema: {os.path.basename(pkl_path)}")
        required_base_keys = ["user_date", "num", "seq", "labels"]
        missing_base = [k for k in required_base_keys if k not in self.data]
        if missing_base:
            raise ValueError(f"Invalid dataset pickle schema: {os.path.basename(pkl_path)}. Missing keys {missing_base}")
        has_ctx = "ctx" in self.data
        has_split_ctx = ("ctx_dynamic" in self.data) and ("ctx_static" in self.data)
        if not (has_ctx or has_split_ctx):
            raise ValueError(f"Invalid dataset pickle schema: {os.path.basename(pkl_path)}. Expected ctx or ctx_dynamic+ctx_static")
        self.num_data = torch.tensor(self.data["num"], dtype=torch.float32)
        if has_ctx:
            self.ctx_data = torch.tensor(self.data["ctx"], dtype=torch.float32)
            self.ctx_dynamic_data = None
            self.ctx_static_data = None
            self.ctx_dynamic_dim = 0
            self.ctx_static_dim = 0
        else:
            ctx_dynamic = torch.tensor(self.data["ctx_dynamic"], dtype=torch.float32)
            ctx_static = torch.tensor(self.data["ctx_static"], dtype=torch.float32)
            if ctx_dynamic.ndim == 1:
                ctx_dynamic = ctx_dynamic.unsqueeze(1)
            if ctx_static.ndim == 1:
                ctx_static = ctx_static.unsqueeze(1)
            if ctx_dynamic.shape[0] != ctx_static.shape[0]:
                raise ValueError("ctx_dynamic and ctx_static must have same number of rows")
            self.ctx_dynamic_data = ctx_dynamic
            self.ctx_static_data = ctx_static
            self.ctx_dynamic_dim = int(ctx_dynamic.shape[1])
            self.ctx_static_dim = int(ctx_static.shape[1])
            self.ctx_data = torch.cat([ctx_dynamic, ctx_static], dim=1)
        self.seq_data = torch.tensor(self.data["seq"], dtype=torch.long)
        self.labels = torch.tensor(self.data["labels"], dtype=torch.long)
        self.user_dates = self.data["user_date"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        item = {
            "num": self.num_data[idx],
            "ctx": self.ctx_data[idx],
            "seq": self.seq_data[idx],
            "label": self.labels[idx],
            "user_date": self.user_dates[idx],
        }
        if self.ctx_dynamic_data is not None and self.ctx_static_data is not None:
            item["ctx_dynamic"] = self.ctx_dynamic_data[idx]
            item["ctx_static"] = self.ctx_static_data[idx]
        return item


def find_latest_checkpoint(model_path: str | None, search_dir: str, filename: str = "model_checkpoint.pt") -> str | None:
    if model_path:
        return model_path
    candidates: list[str] = []
    for root, _, files in os.walk(search_dir):
        if filename in files:
            candidates.append(os.path.join(root, filename))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def save_model_checkpoint(model: nn.Module, meta: dict, output_dir: str, filename: str = "model_checkpoint.pt") -> str:
    os.makedirs(output_dir, exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "meta": meta,
        "model_name": meta.get("model_name", "static_fused_three_z"),
    }
    out_path = os.path.join(output_dir, filename)
    torch.save(ckpt, out_path)
    return out_path


def load_model_checkpoint(ckpt_path: str, device: str | torch.device = "cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt["meta"]
    model_name = ckpt.get("model_name", "static_fused_three_z")
    if str(model_name) != "static_fused_three_z":
        raise ValueError(f"Only static_fused_three_z is supported in this project, got {model_name}")

    state_dict = ckpt["state_dict"]
    k_num = state_dict["proto_layer_num.prototypes"].shape[0]
    k_dyn = state_dict["proto_layer_dyn.prototypes"].shape[0]
    k_seq = state_dict["proto_layer_seq.prototypes"].shape[0]
    meta["num_prototypes"] = {"num": k_num, "dyn": k_dyn, "seq": k_seq}

    model = StaticFusedThreeZAutoencoder(
        num_dim=meta["num_dim"],
        ctx_dim=meta["ctx_dim"],
        vocab_size=meta["vocab_size"],
        max_seq_len=meta["max_seq_len"],
        embed_dim=meta["embed_dim"],
        z_dim=meta["z_dim"],
        num_prototypes=meta["num_prototypes"],
        ctx_dynamic_dim=meta.get("ctx_dynamic_dim"),
        ctx_static_dim=meta.get("ctx_static_dim"),
    ).to(device)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = getattr(incompatible, "missing_keys", []) or []
    unexpected = getattr(incompatible, "unexpected_keys", []) or []
    if missing or unexpected:
        logger.warning("Checkpoint state_dict loaded with strict=False | missing=%d unexpected=%d", len(missing), len(unexpected))
    proto_summary = meta.get("prototype_summary") if isinstance(meta, dict) else None
    if isinstance(proto_summary, dict):
        radii = proto_summary.get("radii")
        if isinstance(radii, dict):
            try:
                model.proto_radii = {k: torch.tensor(v, device=device, dtype=torch.float32) for k, v in radii.items()}
            except Exception:
                model.proto_radii = radii
    return model, meta


def _kmeans_plus_plus_init(x: torch.Tensor, k: int, generator: torch.Generator) -> torch.Tensor:
    n = int(x.size(0))
    first_idx = torch.randint(0, n, (1,), generator=generator, device=x.device).item()
    centers = [x[first_idx]]
    closest_dist_sq = torch.cdist(x, centers[0].unsqueeze(0), p=2).squeeze(1).pow(2)
    for _ in range(1, int(k)):
        probs = closest_dist_sq / closest_dist_sq.sum().clamp_min(1e-12)
        next_idx = torch.multinomial(probs, 1, generator=generator).item()
        next_center = x[next_idx]
        centers.append(next_center)
        dist_sq_new = torch.cdist(x, next_center.unsqueeze(0), p=2).squeeze(1).pow(2)
        closest_dist_sq = torch.minimum(closest_dist_sq, dist_sq_new)
    return torch.stack(centers, dim=0)


def _kmeans_torch(x: torch.Tensor, k: int, num_iters: int = 25, seed: int = 0) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError("kmeans expects x to be 2D (n, d)")
    if int(x.size(0)) < int(k):
        raise ValueError(f"kmeans expects n >= k, got n={x.size(0)} k={k}")
    try:
        generator = torch.Generator(device=x.device)
    except TypeError:
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    centers = _kmeans_plus_plus_init(x, k=int(k), generator=generator)
    for _ in range(int(num_iters)):
        distances = torch.cdist(x, centers, p=2)
        labels = distances.argmin(dim=1)
        d = int(x.size(1))
        new_centers = torch.zeros((int(k), d), device=x.device, dtype=x.dtype)
        new_centers.scatter_add_(0, labels.view(-1, 1).expand(-1, d), x)
        counts = torch.bincount(labels, minlength=int(k)).to(x.device).to(x.dtype)
        new_centers = new_centers / counts.view(-1, 1).clamp_min(1.0)
        empty = counts == 0
        if empty.any():
            min_dist = distances.min(dim=1).values
            farthest = min_dist.argsort(descending=True)
            empty_idx = torch.nonzero(empty, as_tuple=False).view(-1)
            take = farthest[: empty_idx.numel()]
            new_centers[empty_idx] = x[take]
        centers = new_centers
    return centers


def _kmeans_torch_with_labels(x: torch.Tensor, k: int, num_iters: int = 25, seed: int = 0):
    centers = _kmeans_torch(x, k=int(k), num_iters=int(num_iters), seed=int(seed))
    distances = torch.cdist(x, centers, p=2)
    labels = distances.argmin(dim=1)
    counts = torch.bincount(labels, minlength=int(k)).to(device=x.device)
    return centers, labels, counts


def _initialize_prototypes_with_kmeans(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    max_samples: int = 50000,
    num_iters: int = 25,
    seed: int = 0,
    run_dir: str | None = None,
):
    has_multihead = all(hasattr(model, name) for name in ("proto_layer_num", "proto_layer_dyn", "proto_layer_seq", "z_dim"))
    if not has_multihead:
        return None
    model.eval()
    n_collected = 0
    z_chunks: dict[str, list[torch.Tensor]] = {"num": [], "dyn": [], "seq": []}
    with torch.no_grad():
        for batch in train_loader:
            num = batch["num"].to(device)
            seq = batch["seq"].to(device)
            ctx_dynamic = batch.get("ctx_dynamic", None)
            ctx_static = batch.get("ctx_static", None)
            if ctx_dynamic is None or ctx_static is None:
                continue
            ctx_dynamic = ctx_dynamic.to(device)
            ctx_static = ctx_static.to(device)
            outputs = model(num, ctx_dynamic, ctx_static, seq)
            z = outputs[0]
            if z.dim() != 2 or int(z.size(1)) != int(model.z_dim) * 3:
                continue
            z_num_f, z_dyn_f, z_seq_f = z.split(int(model.z_dim), dim=-1)
            z_chunks["num"].append(F.normalize(z_num_f, p=2, dim=1).detach())
            z_chunks["dyn"].append(F.normalize(z_dyn_f, p=2, dim=1).detach())
            z_chunks["seq"].append(F.normalize(z_seq_f, p=2, dim=1).detach())
            n_collected += int(z.size(0))
            if n_collected >= int(max_samples):
                break
    if not all(z_chunks[m] for m in ("num", "dyn", "seq")):
        model.train()
        return None
    
    init_stats: dict = {"k": {}, "max_samples": int(max_samples), "num_iters": int(num_iters), "seed": int(seed), "dims": {}, "totals": {}}
    
    for m, layer_attr, chunks in (
        ("num", "proto_layer_num", z_chunks["num"]),
        ("dyn", "proto_layer_dyn", z_chunks["dyn"]),
        ("seq", "proto_layer_seq", z_chunks["seq"]),
    ):
        # [修复1]: 直接从当前模型对应的 layer 读取其真实的 k，避免字典 k 不一致导致的维度崩溃
        current_k = int(getattr(model, layer_attr).prototypes.size(0))
        init_stats["k"][m] = current_k
        
        z_all = torch.cat(chunks, dim=0)
        if int(z_all.size(0)) > int(max_samples):
            z_all = z_all[: int(max_samples)]
            
        centers, _, counts = _kmeans_torch_with_labels(z_all, k=current_k, num_iters=int(num_iters), seed=int(seed))
        centers = F.normalize(centers, p=2, dim=1)
        with torch.no_grad():
            getattr(model, layer_attr).prototypes.copy_(centers)
            
        init_stats["dims"][m] = counts.detach().cpu().numpy().astype(int).tolist()
        init_stats["totals"][m] = int(z_all.size(0))
        
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        out_csv = os.path.join(run_dir, "kmeans_init_cluster_sizes.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["dim", "k", "cluster_id", "count", "total", "frac", "max_samples", "num_iters", "seed"],
            )
            w.writeheader()
            for dim_key, counts_list in init_stats["dims"].items():
                total = int(init_stats["totals"].get(dim_key, 0))
                k_val = init_stats["k"][dim_key]
                for cid, cnt in enumerate(counts_list):
                    frac = (float(cnt) / float(total)) if total > 0 else 0.0
                    w.writerow(
                        {
                            "dim": str(dim_key),
                            "k": int(k_val),
                            "cluster_id": int(cid),
                            "count": int(cnt),
                            "total": int(total),
                            "frac": float(frac),
                            "max_samples": int(max_samples),
                            "num_iters": int(num_iters),
                            "seed": int(seed),
                        }
                    )
    model.train()
    return init_stats


def train_model(
    train_pkl: str,
    num_epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str | torch.device = "cuda",
    embed_dim: int = 64,
    z_dim: int = 32,
    weights: dict | None = None,
    seperate: int = 0,
    num_prototypes: dict | int | None = {"num": 30, "dyn": 30, "seq": 30},
    prune_ratio: float = 0.01,
    run_dir: str | None = None,
):
    def _prototype_losses(seperate_method: int, min_dist: torch.Tensor, prototypes: torch.Tensor):
        loss_clustering = torch.mean(min_dist)
        if int(prototypes.size(0)) <= 1:
            return loss_clustering, torch.tensor(0.0, device=prototypes.device)
        k = int(prototypes.size(0))
        mask = torch.eye(k, device=prototypes.device).bool()
        dist_matrix = torch.cdist(prototypes, prototypes, p=2)
        off_diag = dist_matrix[~mask]
        if int(seperate_method) == 1:
            margin = 2.0
            loss_separation = torch.mean(F.relu(margin - off_diag))
        elif int(seperate_method) == 2:
            tau = 1.0
            loss_separation = torch.mean(torch.exp(-(off_diag**2) / tau))
        elif int(seperate_method) == 3:
            epsilon = 1e-4
            loss_separation = torch.mean(1.0 / (off_diag + epsilon))
        elif int(seperate_method) == 4:
            masked_dist = dist_matrix.clone()
            masked_dist.fill_diagonal_(float("inf"))
            min_dists_between_protos, _ = torch.min(masked_dist, dim=1)
            margin = 2.0
            loss_separation = torch.mean(F.relu(margin - min_dists_between_protos))
        else:
            loss_separation = torch.mean(torch.exp(-off_diag))
        return loss_clustering, loss_separation

    logger.info("[%s] >>> 初始化数据集...", datetime.now())
    train_dataset = CERTDataset(train_pkl)
    train_loader = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True)

    num_dim = int(train_dataset.num_data.shape[1])
    ctx_dim = int(train_dataset.ctx_data.shape[1])
    ctx_dynamic_dim = int(getattr(train_dataset, "ctx_dynamic_dim", 0) or 0)
    ctx_static_dim = int(getattr(train_dataset, "ctx_static_dim", 0) or 0)
    max_seq_len = int(train_dataset.seq_data.shape[1])
    vocab_size = int(torch.max(train_dataset.seq_data).item() + 1)
    vocab_size = max(vocab_size, 8)

    logger.info("[%s] >>> 初始化模型: static_fused_three_z | Vocab Size: %d", datetime.now(), vocab_size)

    model = StaticFusedThreeZAutoencoder(
        num_dim=num_dim,
        ctx_dim=ctx_dim,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=int(embed_dim),
        z_dim=int(z_dim),
        num_prototypes=num_prototypes, 
        ctx_dynamic_dim=int(ctx_dynamic_dim),
        ctx_static_dim=int(ctx_static_dim),
    ).to(device)

    init_kmeans_stats = None
    optimizer = optim.Adam(model.parameters(), lr=float(lr))
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss(ignore_index=0)

    if weights is None:
        weights = {"num": 1.0, "ctx": 10.0, "seq": 0.1}
    weights = {"num": float(weights.get("num", 1.0)), "ctx": float(weights.get("ctx", 1.0)), "seq": float(weights.get("seq", 0.1))}

    logger.info("[%s] >>> 开始训练...", datetime.now())
    model.train()

    phase1_epochs = max(1, int(num_epochs * 0.4)) 
    kmeans_init_epoch = phase1_epochs 
    stop_revival_epoch = max(1, int(num_epochs * 0.8)) 
    kmeans_inited = False

    train_proto_csv_path = None
    csv_f = None
    csv_w = None
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        train_proto_csv_path = os.path.join(run_dir, "prototype_train_epoch_stats.csv")
        csv_f = open(train_proto_csv_path, "w", newline="", encoding="utf-8")
        csv_w = csv.DictWriter(
            csv_f,
            fieldnames=["epoch", "dim", "k", "delta_fro", "dist_matrix", "hit_counts", "radius", "user_counts", "avg_users"],
        )
        csv_w.writeheader()

    prev_proto_dm: dict[str, torch.Tensor] = {}
    last_epoch_stats = None

    for epoch in range(int(num_epochs)):
        if (not kmeans_inited) and epoch >= kmeans_init_epoch:
            device_obj = device if isinstance(device, torch.device) else torch.device(device)
            init_kmeans_stats = _initialize_prototypes_with_kmeans(
                model=model, train_loader=train_loader, device=device_obj, run_dir=run_dir
            )
            kmeans_inited = isinstance(init_kmeans_stats, dict)
            if kmeans_inited:
                for layer_attr in ("proto_layer_num", "proto_layer_dyn", "proto_layer_seq"):
                    layer = getattr(model, layer_attr, None)
                    if layer and isinstance(layer.prototypes, torch.nn.Parameter) and layer.prototypes in optimizer.state:
                        optimizer.state[layer.prototypes].clear()
                logger.info("[%s] >>> [阶段二开启] Prototype K-means 初始化完成 (epoch=%d)", datetime.now(), int(epoch))

        compute_proto_loss = kmeans_inited 
        allow_revival = compute_proto_loss and (epoch < stop_revival_epoch)

        total_loss = 0.0
        total_num_loss = 0.0
        total_ctx_loss = 0.0
        total_seq_loss = 0.0
        total_proto_clustering = 0.0
        total_proto_separation = 0.0

        for batch in train_loader:
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

            optimizer.zero_grad()
            outputs = model(num, ctx_dynamic, ctx_static, seq)
            z_cat = outputs[0]
            r_num = outputs[1]
            r_ctx = outputs[2]
            r_seq = outputs[3]
            proto_dists = outputs[4]

            loss_num = criterion_mse(r_num, num)
            loss_ctx = torch.tensor(0.0, device=num.device)
            if torch.is_tensor(r_ctx) and r_ctx.shape == ctx.shape:
                loss_ctx = criterion_mse(r_ctx, ctx)
            loss_seq = criterion_ce(r_seq.view(-1, vocab_size), seq.view(-1))

            loss = weights["num"] * loss_num + weights["ctx"] * loss_ctx + weights["seq"] * loss_seq

            loss_clustering_sum = torch.tensor(0.0, device=num.device)
            loss_separation_sum = torch.tensor(0.0, device=num.device)
            if compute_proto_loss:
                lambda_1 = 10.0 
                lambda_2 = 0.01
                for m, layer_attr in (("num", "proto_layer_num"), ("dyn", "proto_layer_dyn"), ("seq", "proto_layer_seq")):
                    min_dist = proto_dists[m]
                    prototypes = getattr(model, layer_attr).prototypes
                    loss_clustering, loss_separation = _prototype_losses(seperate_method=int(seperate), min_dist=min_dist, prototypes=prototypes)
                    loss_clustering_sum = loss_clustering_sum + loss_clustering
                    loss_separation_sum = loss_separation_sum + loss_separation
                loss = loss + lambda_1 * loss_clustering_sum + lambda_2 * loss_separation_sum

            loss.backward()
            optimizer.step()

            if allow_revival:
                z_num_f, z_dyn_f, z_seq_f = z_cat.split(int(z_dim), dim=-1)
                z_feats = {"num": z_num_f, "dyn": z_dyn_f, "seq": z_seq_f}
                for m, layer_attr in (("num", "proto_layer_num"), ("dyn", "proto_layer_dyn"), ("seq", "proto_layer_seq")):
                    prototypes = getattr(model, layer_attr).prototypes
                    norm_protos = F.normalize(prototypes, p=2, dim=1)
                    norm_z = F.normalize(z_feats[m].detach(), p=2, dim=1)
                    dist = torch.cdist(norm_z, norm_protos, p=2)
                    _, hit_ids = torch.min(dist, dim=1)
                    
                    hit_counts_batch = torch.bincount(hit_ids, minlength=prototypes.size(0))
                    dead_mask = hit_counts_batch == 0
                    if dead_mask.any():
                        dead_indices = torch.nonzero(dead_mask, as_tuple=False).view(-1)
                        for d_idx in dead_indices:
                            rand_idx = torch.randint(0, norm_z.size(0), (1,)).item()
                            with torch.no_grad():
                                prototypes[d_idx].copy_(norm_z[rand_idx])

            total_loss += float(loss.item())
            total_num_loss += float(loss_num.item())
            total_ctx_loss += float(loss_ctx.item())
            total_seq_loss += float(loss_seq.item())
            total_proto_clustering += float(loss_clustering_sum.item())
            total_proto_separation += float(loss_separation_sum.item())

        if epoch == stop_revival_epoch:
            logger.info("[%s] >>> [稳定期开启] 原型复活机制已关闭，准备最终流形收敛和剪枝 (epoch=%d)", datetime.now(), int(epoch))

        avg_loss = total_loss / max(1, len(train_loader))
        msg = (
            f"Epoch [{epoch + 1}/{num_epochs}] | Loss: {avg_loss:.4f} | "
            f"Num(MSE): {total_num_loss/len(train_loader):.4f} | "
            f"Ctx(MSE): {total_ctx_loss/len(train_loader):.4f} | "
            f"Seq(CE): {total_seq_loss/len(train_loader):.4f} | "
            f"Proto(Cluster): {total_proto_clustering/len(train_loader):.4f} | "
            f"Proto(Sep): {total_proto_separation/len(train_loader):.4f}"
        )
        logger.info(msg)

        if csv_w is not None:
            model.eval()
            
            # [修复2]: 不再使用全局单一的 k，而是依据各个层真实的 k 动态创建字典结构
            k_dict = {
                "num": model.proto_layer_num.prototypes.size(0),
                "dyn": model.proto_layer_dyn.prototypes.size(0),
                "seq": model.proto_layer_seq.prototypes.size(0),
            }
            
            hit_counts = {d: torch.zeros((k_dict[d],), device=device, dtype=torch.long) for d in ("num", "dyn", "seq")}
            radius = {d: torch.zeros((k_dict[d],), device=device, dtype=torch.float32) for d in ("num", "dyn", "seq")}
            user_sets = {d: [set() for _ in range(k_dict[d])] for d in ("num", "dyn", "seq")}

            proto_num = F.normalize(model.proto_layer_num.prototypes, p=2, dim=1)
            proto_dyn = F.normalize(model.proto_layer_dyn.prototypes, p=2, dim=1)
            proto_seq = F.normalize(model.proto_layer_seq.prototypes, p=2, dim=1)

            with torch.no_grad():
                for batch in train_loader:
                    num = batch["num"].to(device)
                    seq = batch["seq"].to(device)
                    ctx_dynamic = batch.get("ctx_dynamic", None)
                    ctx_static = batch.get("ctx_static", None)
                    if ctx_dynamic is None or ctx_static is None:
                        continue
                    ctx_dynamic = ctx_dynamic.to(device)
                    ctx_static = ctx_static.to(device)
                    outputs = model(num, ctx_dynamic, ctx_static, seq)
                    z_cat = outputs[0]
                    if z_cat.dim() != 2 or int(z_cat.size(1)) != int(model.z_dim) * 3:
                        continue
                    z_num_f, z_dyn_f, z_seq_f = z_cat.split(int(model.z_dim), dim=-1)
                    z_proto_num = F.normalize(z_num_f, p=2, dim=1)
                    z_proto_dyn = F.normalize(z_dyn_f, p=2, dim=1)
                    z_proto_seq = F.normalize(z_seq_f, p=2, dim=1)

                    user_list = [str(x).rsplit("_", 1)[0] if "_" in str(x) else str(x) for x in batch.get("user_date", [])]
                    if len(user_list) != int(z_cat.size(0)):
                        user_list = [""] * int(z_cat.size(0))

                    batch_pack = {"num": (z_proto_num, proto_num), "dyn": (z_proto_dyn, proto_dyn), "seq": (z_proto_seq, proto_seq)}
                    for dim_key, (z_proto_dim, proto_dim) in batch_pack.items():
                        current_k = k_dict[dim_key] # 使用当前分支的 k
                        d = torch.cdist(z_proto_dim, proto_dim, p=2)
                        min_dist, hit_id = torch.min(d, dim=1)
                        hit_counts[dim_key] = hit_counts[dim_key] + torch.bincount(hit_id, minlength=current_k).to(device=device, dtype=torch.long)
                        for j in range(current_k):
                            mask = hit_id == int(j)
                            if mask.any():
                                radius[dim_key][j] = torch.maximum(radius[dim_key][j], min_dist[mask].max())
                                for u in (user_list[i] for i in torch.nonzero(mask, as_tuple=False).view(-1).tolist()):
                                    if u:
                                        user_sets[dim_key][j].add(u)

            for dim_key, proto_dim in (("num", proto_num), ("dyn", proto_dyn), ("seq", proto_seq)):
                current_k = k_dict[dim_key]
                dm = torch.cdist(proto_dim, proto_dim, p=2)
                prev = prev_proto_dm.get(dim_key, None)
                delta = 0.0
                if prev is not None and prev.shape == dm.shape:
                    delta = float(torch.norm(dm - prev, p="fro").detach().cpu().item())
                prev_proto_dm[dim_key] = dm.detach()
                user_counts = [len(s) for s in user_sets[dim_key]]
                avg_users = float(sum(user_counts) / max(1, len(user_counts)))
                csv_w.writerow(
                    {
                        "epoch": int(epoch + 1),
                        "dim": str(dim_key),
                        "k": int(current_k),
                        "delta_fro": float(delta),
                        "dist_matrix": dm.detach().cpu().numpy().tolist(),
                        "hit_counts": hit_counts[dim_key].detach().cpu().numpy().astype(int).tolist(),
                        "radius": radius[dim_key].detach().cpu().numpy().astype(float).tolist(),
                        "user_counts": user_counts,
                        "avg_users": float(avg_users),
                    }
                )

            radii_list = {d: radius[d].detach().cpu().numpy().astype(float).tolist() for d in ("num", "dyn", "seq")}
            hit_list = {d: hit_counts[d].detach().cpu().numpy().astype(int).tolist() for d in ("num", "dyn", "seq")}
            user_count_list = {d: [len(s) for s in user_sets[d]] for d in ("num", "dyn", "seq")}
            last_epoch_stats = {"k": k_dict, "radii": radii_list, "hit_counts": hit_list, "user_counts": user_count_list}
            model.proto_radii = {d: torch.tensor(radii_list[d], device=device, dtype=torch.float32) for d in ("num", "dyn", "seq")}
            model.train()

    logger.info("[%s] >>> 开始原型剪枝 (Threshold Ratio: %.2f%%)", datetime.now(), prune_ratio * 100)
    
    min_hits_required = max(1, int(len(train_loader.dataset) * prune_ratio))
    pruned_k_dict = {}

    for dim_key, layer_attr in (("num", "proto_layer_num"), ("dyn", "proto_layer_dyn"), ("seq", "proto_layer_seq")):
        hits = last_epoch_stats["hit_counts"][dim_key] 
        hits_tensor = torch.tensor(hits, device=device)
        
        keep_mask = hits_tensor >= min_hits_required
        
        if not keep_mask.any():
            keep_mask[torch.argmax(hits_tensor)] = True
            
        layer = getattr(model, layer_attr)
        old_k = int(layer.prototypes.size(0))
        new_k = int(keep_mask.sum().item())
        
        if new_k < old_k:
            with torch.no_grad():
                kept_protos = layer.prototypes[keep_mask].clone()
                layer.prototypes = nn.Parameter(kept_protos)
                
                if hasattr(model, 'proto_radii') and dim_key in model.proto_radii:
                    model.proto_radii[dim_key] = model.proto_radii[dim_key][keep_mask]
                    
        pruned_k_dict[dim_key] = new_k
        logger.info(f" -> [{dim_key}] 剪枝完成: {old_k} -> {new_k} 个 (存活线: {min_hits_required} 命中)")
        
        if isinstance(last_epoch_stats, dict):
            last_epoch_stats["k"] = pruned_k_dict
            last_epoch_stats["radii"][dim_key] = [r for i, r in enumerate(last_epoch_stats["radii"][dim_key]) if keep_mask[i]]
            last_epoch_stats["hit_counts"][dim_key] = [h for i, h in enumerate(last_epoch_stats["hit_counts"][dim_key]) if keep_mask[i]]
            last_epoch_stats["user_counts"][dim_key] = [u for i, u in enumerate(last_epoch_stats["user_counts"][dim_key]) if keep_mask[i]]
    
    if csv_f is not None:
        csv_f.flush()
        csv_f.close()

    proto_summary = None
    if isinstance(last_epoch_stats, dict):
        proto_summary = {
            "k": last_epoch_stats["k"],
            "radii": last_epoch_stats["radii"],
            "hit_counts": last_epoch_stats["hit_counts"],
            "user_counts": last_epoch_stats["user_counts"],
            "train_proto_csv": train_proto_csv_path,
        }

    meta = {
        "model_name": "static_fused_three_z",
        "num_dim": int(num_dim),
        "ctx_dim": int(ctx_dim),
        "ctx_dynamic_dim": int(ctx_dynamic_dim),
        "ctx_static_dim": int(ctx_static_dim),
        "max_seq_len": int(max_seq_len),
        "vocab_size": int(vocab_size),
        "embed_dim": int(embed_dim),
        "z_dim": int(z_dim),
        "weights": weights,
        "seperate": int(seperate),
        "num_prototypes": pruned_k_dict,
    }
    if isinstance(init_kmeans_stats, dict):
        meta["prototype_init_kmeans"] = init_kmeans_stats
    if proto_summary is not None:
        meta["prototype_summary"] = proto_summary
    return model, meta