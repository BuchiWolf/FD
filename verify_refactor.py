import torch
from models import OptimizedThreeZAutoencoder
from evaluation import evaluate_and_extract_z_optimized
from training import finetune_model_with_buffer_optimized
import numpy as np
import os
import pickle

def test_workflow():
    print(">>> 启动重构代码逻辑验证...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 模拟数据
    num_dim, ctx_static_dim, ctx_dynamic_dim = 10, 5, 5
    vocab_size, max_seq_len = 10, 20
    batch_size = 4
    num_samples = 100
    
    data = {
        "num": torch.randn(num_samples, num_dim),
        "ctx_dynamic": torch.randn(num_samples, ctx_dynamic_dim),
        "ctx_static": torch.randn(num_samples, ctx_static_dim),
        "seq": torch.randint(1, vocab_size, (num_samples, max_seq_len)),
        "labels": torch.zeros(num_samples).long(),
        "user_date": [f"user_{i}" for i in range(num_samples)]
    }
    data["labels"][90:] = 1 # 模拟异常
    
    test_pkl = "test_mock.pkl"
    with open(test_pkl, "wb") as f:
        pickle.dump(data, f)
    
    # 2. 初始化模型
    model = OptimizedThreeZAutoencoder(
        num_dim=num_dim, ctx_dim=ctx_static_dim + ctx_dynamic_dim,
        vocab_size=vocab_size, max_seq_len=max_seq_len,
        z_dim=16, num_prototypes=10,
        ctx_dynamic_dim=ctx_dynamic_dim, ctx_static_dim=ctx_static_dim
    ).to(device)
    
    print(">>> 模型初始化成功.")
    
    # 3. 验证评估函数
    weights = {"num": 1.0, "ctx": 1.0, "seq": 0.1}
    results = evaluate_and_extract_z_optimized(
        model, test_pkl, batch_size=batch_size, device=device, weights=weights,
        vocab_size=vocab_size, enable_tta=True, low_score_anomaly_dbscan_min_samples=5
    )
    
    print(f">>> 评估验证完成: AUC={results['auc']:.4f}, PR-AUC={results['pr_auc']:.4f}")
    
    # 4. 验证微调函数
    memory_buffer = {
        "num": data["num"][:20],
        "ctx_dynamic": data["ctx_dynamic"][:20],
        "ctx_static": data["ctx_static"][:20],
        "seq": data["seq"][:20]
    }
    new_cluster_data = {
        "num": data["num"][20:30],
        "ctx_dynamic": data["ctx_dynamic"][20:30],
        "ctx_static": data["ctx_static"][20:30],
        "seq": data["seq"][20:30]
    }
    
    model, updated_buffer = finetune_model_with_buffer_optimized(
        model, memory_buffer, new_cluster_data, device, vocab_size, weights
    )
    
    print(">>> 微调验证完成.")
    print(">>> 所有核心模块逻辑验证通过！")
    
    # 清理
    if os.path.exists(test_pkl):
        os.remove(test_pkl)

if __name__ == "__main__":
    test_workflow()
