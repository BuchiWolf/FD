import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeLayer(nn.Module):
    def __init__(self, k: int, latent_dim: int):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(int(k), int(latent_dim)))

    def forward(self, z: torch.Tensor):
        normalized_prototypes = F.normalize(self.prototypes, p=2, dim=1)
        distances = torch.cdist(z, normalized_prototypes, p=2)
        min_dist, _ = torch.min(distances, dim=1)
        return min_dist, normalized_prototypes


class TransformerStaticFusion(nn.Module):
    def __init__(self, z_dim: int, ctx_static_dim: int, embed_dim: int = 64):
        super().__init__()
        self.z_proj = nn.Linear(int(z_dim), int(embed_dim))
        self.ctx_proj = nn.Linear(int(ctx_static_dim), int(embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=int(embed_dim), nhead=4, batch_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.out_proj = nn.Linear(int(embed_dim), int(z_dim))

    def forward(self, z: torch.Tensor, ctx_static: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([self.z_proj(z), self.ctx_proj(ctx_static)], dim=1)
        fused = self.enc(tokens)
        return self.out_proj(fused[:, 0])


class MLPStaticFusion(nn.Module):
    def __init__(self, z_dim: int, ctx_static_dim: int, embed_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(int(z_dim) + int(ctx_static_dim), int(embed_dim)),
            nn.ReLU(),
            nn.Linear(int(embed_dim), int(z_dim))
        )

    def forward(self, z: torch.Tensor, ctx_static: torch.Tensor) -> torch.Tensor:
        fused_input = torch.cat([z, ctx_static], dim=-1)
        return self.mlp(fused_input)


class StaticFusedThreeZAutoencoder(nn.Module):
    def __init__(
        self,
        num_dim: int,
        ctx_dim: int,
        vocab_size: int,
        max_seq_len: int,
        embed_dim: int = 64,
        z_dim: int = 32,
        num_prototypes: dict | int | None = {"num": 10, "dyn": 10, "seq": 10},
        ctx_dynamic_dim: int | None = None,
        ctx_static_dim: int | None = None,
    ):
        super().__init__()
        self.max_seq_len = int(max_seq_len)
        self.vocab_size = int(vocab_size)
        self.z_dim = int(z_dim)

        if ctx_dynamic_dim is None and ctx_static_dim is None:
            raise ValueError("StaticFusedThreeZAutoencoder requires ctx_dynamic_dim and ctx_static_dim.")
        if ctx_dynamic_dim is None:
            ctx_dynamic_dim = int(ctx_dim) - int(ctx_static_dim)
        if ctx_static_dim is None:
            ctx_static_dim = int(ctx_dim) - int(ctx_dynamic_dim)
        if int(ctx_dynamic_dim) <= 0 or int(ctx_static_dim) <= 0:
            raise ValueError("Invalid ctx split.")
        self.ctx_dynamic_dim = int(ctx_dynamic_dim)
        self.ctx_static_dim = int(ctx_static_dim)

        self.num_enc = nn.Sequential(nn.Linear(int(num_dim), int(embed_dim)), nn.ReLU(), nn.Linear(int(embed_dim), int(embed_dim)))
        self.ctx_dyn_enc = nn.Sequential(
            nn.Linear(int(self.ctx_dynamic_dim), int(embed_dim)),
            nn.ReLU(),
            nn.Linear(int(embed_dim), int(embed_dim)),
        )

        self.seq_emb = nn.Embedding(num_embeddings=int(vocab_size), embedding_dim=int(embed_dim), padding_idx=0)
        self.seq_pos_emb = nn.Embedding(num_embeddings=int(max_seq_len), embedding_dim=int(embed_dim))
        seq_encoder_layer = nn.TransformerEncoderLayer(d_model=int(embed_dim), nhead=4, batch_first=True)
        self.seq_enc = nn.TransformerEncoder(seq_encoder_layer, num_layers=1)

        self.shared_encoder = nn.Sequential(nn.Linear(int(embed_dim) * 3, 128), nn.ReLU(), nn.Linear(128, int(z_dim) * 3))
        self.static_fuse = MLPStaticFusion(z_dim=int(z_dim), ctx_static_dim=int(self.ctx_static_dim), embed_dim=int(embed_dim))
        
        if isinstance(num_prototypes, dict):
            k_num = num_prototypes.get("num", 10)
            k_dyn = num_prototypes.get("dyn", 10)
            k_seq = num_prototypes.get("seq", 10)
        else:
            k_num = k_dyn = k_seq = int(num_prototypes)

        self.proto_layer_num = PrototypeLayer(k=k_num, latent_dim=int(z_dim))
        self.proto_layer_dyn = PrototypeLayer(k=k_dyn, latent_dim=int(z_dim))
        self.proto_layer_seq = PrototypeLayer(k=k_seq, latent_dim=int(z_dim))

        self.num_dec = nn.Sequential(nn.Linear(int(z_dim), int(embed_dim)), nn.ReLU(), nn.Linear(int(embed_dim), int(num_dim)))
        self.ctx_dyn_dec = nn.Sequential(
            nn.Linear(int(z_dim), int(embed_dim)),
            nn.ReLU(),
            nn.Linear(int(embed_dim), int(self.ctx_dynamic_dim)),
        )
        self.ctx_static_dec = nn.Sequential(
            nn.Linear(int(z_dim), int(embed_dim)),
            nn.ReLU(),
            nn.Linear(int(embed_dim), int(self.ctx_static_dim)),
        )
        self.seq_dec_lstm = nn.LSTM(input_size=int(embed_dim + z_dim), hidden_size=int(embed_dim), batch_first=True)
        self.seq_out = nn.Linear(int(embed_dim), int(vocab_size))

    def forward(
        self,
        num: torch.Tensor,
        ctx_dynamic: torch.Tensor,
        ctx_static: torch.Tensor,
        seq: torch.Tensor,
    ):
        e_num = self.num_enc(num)
        e_dyn = self.ctx_dyn_enc(ctx_dynamic)

        seq_len = int(seq.size(1))
        pos_ids = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(seq.size(0), seq_len)
        e_seq_in = self.seq_emb(seq) + self.seq_pos_emb(pos_ids)
        key_padding_mask = seq.eq(0)
        e_seq_all = self.seq_enc(e_seq_in, src_key_padding_mask=key_padding_mask)
        valid_mask = (~key_padding_mask).unsqueeze(-1)
        e_seq_sum = (e_seq_all * valid_mask).sum(dim=1)
        denom = valid_mask.sum(dim=1).clamp(min=1)
        e_seq = e_seq_sum / denom

        concat_e = torch.cat([e_num, e_dyn, e_seq], dim=-1)
        z_all = self.shared_encoder(concat_e)
        z_num, z_dyn, z_seq = z_all.split(self.z_dim, dim=-1)
        
        z_num = F.normalize(z_num, p=2, dim=1)
        z_dyn = F.normalize(z_dyn, p=2, dim=1)
        z_seq = F.normalize(z_seq, p=2, dim=1)

        z_num_f = F.normalize(self.static_fuse(z_num, ctx_static), p=2, dim=1)
        z_dyn_f = F.normalize(self.static_fuse(z_dyn, ctx_static), p=2, dim=1)
        z_seq_f = F.normalize(self.static_fuse(z_seq, ctx_static), p=2, dim=1)

        z = (z_num_f + z_dyn_f + z_seq_f) / 3.0
        z = F.normalize(z, p=2, dim=1)

        z_cat = torch.cat([z_num_f, z_dyn_f, z_seq_f], dim=-1)

        min_dist_num, norm_proto_num = self.proto_layer_num(F.normalize(z_num_f, p=2, dim=1))
        min_dist_dyn, norm_proto_dyn = self.proto_layer_dyn(F.normalize(z_dyn_f, p=2, dim=1))
        min_dist_seq, norm_proto_seq = self.proto_layer_seq(F.normalize(z_seq_f, p=2, dim=1))
        
        proto_dists = {"num": min_dist_num, "dyn": min_dist_dyn, "seq": min_dist_seq}
        proto_norms = {"num": norm_proto_num, "dyn": norm_proto_dyn, "seq": norm_proto_seq}

        r_num = self.num_dec(z_num_f)
        r_ctx_dynamic = self.ctx_dyn_dec(z_dyn_f)
        r_ctx_static = self.ctx_static_dec(z)
        r_ctx = torch.cat([r_ctx_dynamic, r_ctx_static], dim=1)

        batch_size = seq.size(0)
        sos_tokens = torch.full((batch_size, 1), fill_value=1, dtype=seq.dtype, device=seq.device)
        shifted_seq = torch.cat([sos_tokens, seq[:, :-1]], dim=1)
        e_seq_dec = self.seq_emb(shifted_seq) # [batch, seq_len, embed_dim]

        # 将 Z 特征与每一时刻的输入 Token Embedding 拼接
        z_repeated = z_seq_f.unsqueeze(1).repeat(1, seq_len, 1)
        dec_input = torch.cat([e_seq_dec, z_repeated], dim=-1) # [batch, seq_len, embed_dim + z_dim]
        
        r_seq_lstm_out, _ = self.seq_dec_lstm(dec_input)
        r_seq = self.seq_out(r_seq_lstm_out)
        
        return z_cat, r_num, r_ctx, r_seq, proto_dists, proto_norms
