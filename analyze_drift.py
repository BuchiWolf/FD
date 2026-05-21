import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 导入您项目中的数据集
from training import CERTDataset
from torch.utils.data import DataLoader

def extract_features_and_split(model, test_pkl_list, device="cuda", batch_size=256):
    """
    按时间顺序提取特征，并拆分为 num, dyn, seq 三个模态的独立 Z 向量
    """
    model.eval()
    all_z_cat = []
    all_labels = []
    all_time_steps = []
    all_user_dates = []  # 新增：记录每个样本的具体用户和时间标识
    
    with torch.no_grad():
        for time_idx, pkl_path in enumerate(test_pkl_list):
            if not os.path.exists(pkl_path):
                continue
                
            dataset = CERTDataset(pkl_path)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            for batch in loader:
                num = batch["num"].to(device)
                seq = batch["seq"].to(device)
                ctx_dyn = batch.get("ctx_dynamic").to(device)
                ctx_stat = batch.get("ctx_static").to(device)
                
                outputs = model(num, ctx_dyn, ctx_stat, seq)
                z_cat = outputs[0].cpu().numpy() 
                
                all_z_cat.append(z_cat)
                all_labels.extend(batch["label"].numpy())
                all_time_steps.extend([time_idx] * z_cat.shape[0])
                
                # 提取 user_date 用于逐条打印
                user_dates = batch.get("user_date", [])
                if len(user_dates) > 0:
                    all_user_dates.extend(user_dates)
                else:
                    all_user_dates.extend(["Unknown"] * z_cat.shape[0])
                
    z_cat_full = np.concatenate(all_z_cat, axis=0)
    labels_full = np.array(all_labels)
    time_full = np.array(all_time_steps)
    user_dates_full = np.array(all_user_dates)
    
    # 按照 z_dim 拆分三个模态
    z_dim = z_cat_full.shape[1] // 3
    z_num = z_cat_full[:, :z_dim]
    z_dyn = z_cat_full[:, z_dim:2*z_dim]
    z_seq = z_cat_full[:, 2*z_dim:]
    
    return z_cat_full, z_num, z_dyn, z_seq, labels_full, time_full, user_dates_full

def analyze_detailed_drift(z_cat, z_num, z_dyn, z_seq, labels, time_steps, user_dates, test_pkl_list):
    """
    详细分析每个恶意样本的漂移距离，并输出范围（最大值、最小值）
    """
    unique_times = sorted(np.unique(time_steps))
    
    # 用于记录上一个时间步的良性基线质心，防止某个月没有良性数据
    prev_b_cent, prev_b_num, prev_b_dyn, prev_b_seq = None, None, None, None

    print("\n" + "="*60)
    print("🚀 恶意样本深度漂移分析 (Max/Min Range & 明细)")
    print("   * 漂移距离(Deviation) = 该样本到当前月份良性基线(正常行为质心)的欧氏距离")
    print("="*60)

    for t in unique_times:
        pkl_name = os.path.basename(test_pkl_list[t]) if t < len(test_pkl_list) else f"Time_{t}"
        print(f"\n[{pkl_name}]")
        
        mask_t = (time_steps == t)
        mask_benign = mask_t & (labels == 0)
        mask_malicious = mask_t & (labels == 1)
        
        # 1. 计算当前月份的良性基线 (正常员工的平均行为质心)
        if np.any(mask_benign):
            b_cent = np.mean(z_cat[mask_benign], axis=0)
            b_num = np.mean(z_num[mask_benign], axis=0)
            b_dyn = np.mean(z_dyn[mask_benign], axis=0)
            b_seq = np.mean(z_seq[mask_benign], axis=0)
            
            prev_b_cent, prev_b_num, prev_b_dyn, prev_b_seq = b_cent, b_num, b_dyn, b_seq
            
            # 顺便计算一下良性样本的内部波动范围
            b_dists = np.linalg.norm(z_cat[mask_benign] - b_cent, axis=1)
            print(f"  🟢 良性数据 ({np.sum(mask_benign)}条) -> 内部离散度: 均值 {np.mean(b_dists):.4f} | 范围 [{np.min(b_dists):.4f} ~ {np.max(b_dists):.4f}]")
        else:
            b_cent, b_num, b_dyn, b_seq = prev_b_cent, prev_b_num, prev_b_dyn, prev_b_seq
            print(f"  🟢 良性数据 -> 该月无良性数据，使用上一周期的良性基线。")

        if b_cent is None:
            print("  ⚠️ 无法找到良性基线进行比较。")
            continue

        # 2. 处理恶意样本，逐个计算偏离度，并求取最大值、最小值
        if np.any(mask_malicious):
            m_z_cat = z_cat[mask_malicious]
            m_z_num = z_num[mask_malicious]
            m_z_dyn = z_dyn[mask_malicious]
            m_z_seq = z_seq[mask_malicious]
            m_users = user_dates[mask_malicious]

            # 计算所有恶意样本到良性基线的距离
            dists_global = np.linalg.norm(m_z_cat - b_cent, axis=1)
            dists_num = np.linalg.norm(m_z_num - b_num, axis=1)
            dists_dyn = np.linalg.norm(m_z_dyn - b_dyn, axis=1)
            dists_seq = np.linalg.norm(m_z_seq - b_seq, axis=1)

            # 输出恶意样本的汇总范围 (Max/Min)
            print(f"  🔴 恶意数据 ({len(m_users)}条) -> 漂移汇总:")
            print(f"       [全局漂移] 均值: {np.mean(dists_global):.4f} | 范围 [Min: {np.min(dists_global):.4f}  ~  Max: {np.max(dists_global):.4f}]")
            print(f"       [Num漂移] 均值: {np.mean(dists_num):.4f} | 范围 [Min: {np.min(dists_num):.4f}  ~  Max: {np.max(dists_num):.4f}]")
            print(f"       [Dyn漂移] 均值: {np.mean(dists_dyn):.4f} | 范围 [Min: {np.min(dists_dyn):.4f}  ~  Max: {np.max(dists_dyn):.4f}]")
            print(f"       [Seq漂移] 均值: {np.mean(dists_seq):.4f} | 范围 [Min: {np.min(dists_seq):.4f}  ~  Max: {np.max(dists_seq):.4f}]")
            
            # 输出每一个恶意样本的具体明细
            print("  🔎 恶意样本明细列表:")
            # 将恶意样本按全局漂移距离降序排列（最异常的排在前面）
            sorted_indices = np.argsort(-dists_global)
            
            for rank, i in enumerate(sorted_indices):
                print(f"       #{rank+1} [{m_users[i]}] -> 全局偏离: {dists_global[i]:.4f} (Num: {dists_num[i]:.4f}, Dyn: {dists_dyn[i]:.4f}, Seq: {dists_seq[i]:.4f})")
        else:
            print("  🔴 恶意数据 -> 该时间段无恶意样本。")

def main():
    model_path = "/workspace/wangyixin/models/FD/static_fused_three_z_修复空间坍缩与TTA聚类_20260519_162139_990915/model_checkpoint.pt" # 您的模型路径
    data_dir = "/workspace/wangyixin/datasets/CERT/FD-ITDD" # 您的数据路径
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 自动获取测试集
    from main import get_ordered_test_pkls
    test_pkls = get_ordered_test_pkls(data_dir)
    
    from training import load_model_checkpoint
    print(">>> 正在加载模型...")
    model, meta = load_model_checkpoint(model_path, device=device)
    
    print(">>> 正在提取特征与用户信息...")
    z_cat, z_num, z_dyn, z_seq, labels, time_steps, user_dates = extract_features_and_split(model, test_pkls, device=device)
    
    # 执行明细漂移分析
    analyze_detailed_drift(z_cat, z_num, z_dyn, z_seq, labels, time_steps, user_dates, test_pkls)

if __name__ == "__main__":
    main()