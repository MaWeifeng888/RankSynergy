import torch
import torch.nn as nn
import torch.nn.functional as F

from model_few import Feature_embedding_network


class SupportGuidedHyperloopBlock(nn.Module):
    """
    Support-guided loop-wise adaptation block.

    query_h:       [B, Q, D]
    support_proto: [B, D]
    output:        [B, Q, D]
    """
    def __init__(self, dim=512, n_heads=8, dropout=0.10, n_loops=4):
        super().__init__()
        self.dim = dim
        self.n_loops = n_loops

        self.shared_block = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, query_h, support_proto):
        B, Q, D = query_h.shape
        support_token = support_proto.unsqueeze(1)  # [B, 1, D]

        h = query_h
        for _ in range(self.n_loops):
            tokens = torch.cat([support_token, h], dim=1)       # [B, 1+Q, D]
            updated = self.shared_block(tokens)[:, 1:, :]       # [B, Q, D]

            support_expand = support_proto.unsqueeze(1).expand(-1, Q, -1)
            g = self.gate(torch.cat([h, support_expand], dim=-1))

            h = g * updated + (1.0 - g) * h
            h = self.norm(h)

        return h


class HyperSynergy(nn.Module):
    """
    Drop-in replacement for HyperSynergy's original few-shot predictor.

    It keeps the original Feature_embedding_network, so the pretrained
    representation model can be loaded as before.
    """
    def __init__(
        self,
        num_support,
        num_query,
        embed_dim=512,
        hidden_dim=256,
        n_loops=4,
        n_heads=8,
        dropout=0.10,
    ):
        super().__init__()

        self.extractor = Feature_embedding_network()

        self.num_support = num_support
        self.num_query = num_query
        self.embed_dim = embed_dim

        # Kept only for compatibility with the original training script.
        self.inner_l_rate = nn.Parameter(torch.FloatTensor([0.1]), requires_grad=True)

        self.support_encoder = nn.Sequential(
            nn.Linear(embed_dim + 1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.hyperloop = SupportGuidedHyperloopBlock(
            dim=embed_dim,
            n_heads=n_heads,
            dropout=dropout,
            n_loops=n_loops,
        )

        self.pred_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def build_support_prototype(self, support_embeddings, support_labels):
        support_labels = support_labels.unsqueeze(-1).float()  # [B, S, 1]
        support_input = torch.cat([support_embeddings, support_labels], dim=-1)
        support_context = self.support_encoder(support_input)
        support_proto = support_context.mean(dim=1)
        return support_proto

    def run_batch(self, data_train, data0_train, batch_size, train1=True):
        NS = self.num_support
        NB = batch_size

        y = data_train["y"].view(NB, -1).float()
        support_labels = y[:, :NS]
        target_label = y[:, NS:]

        embeddings = self.extractor(data_train, data0_train, NB)
        support_embeddings = embeddings[:, :NS, :]
        target_embeddings = embeddings[:, NS:, :]

        support_proto = self.build_support_prototype(support_embeddings, support_labels)
        refined_query = self.hyperloop(target_embeddings, support_proto)

        query_output = self.pred_head(refined_query).squeeze(-1)
        query_loss = F.mse_loss(query_output, target_label)

        zero = torch.zeros((), device=query_loss.device)
        kl_div = zero
        encoder_penalty = zero
        total_loss = query_loss

        return total_loss, kl_div, encoder_penalty, query_loss, query_output, target_label
