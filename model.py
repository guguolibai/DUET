"""
Model definition: DUET network
- Attribute encoder: input X [N, D]
- Structure encoder: input A [N, N]
- Symmetric cross-attention fusion module
- Classification heads for entrusted teaching
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# Activation function mapping
ACTIVATIONS = {
    "ident": nn.Identity,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.2),
    "prelu": nn.PReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "softmax": lambda: nn.Softmax(dim=1),
}


class SymmetricCrossAttentionFusion(nn.Module):
    """
    Symmetric cross-attention fusion module

    Bidirectional information exchange:
    1. Attribute -> Structure: attribute as query, extract info from structure
    2. Structure -> Attribute: structure as query, extract info from attribute
    3. Fuse results from both directions
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        # Bidirectional cross-attention
        self.attn_a2s = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_s2a = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim)
        )

    def forward(self, z_attr, z_stru):
        # Expand dims: [N, D] -> [N, 1, D]
        a, s = z_attr.unsqueeze(1), z_stru.unsqueeze(1)

        # Cross-attention
        enhanced_a = self.attn_a2s(a, s, s)[0].squeeze(1)  # Attribute extracts from structure
        enhanced_s = self.attn_s2a(s, a, a)[0].squeeze(1)  # Structure extracts from attribute

        # Concatenate and fuse
        combined = torch.cat([enhanced_a, enhanced_s], dim=1)
        return self.fusion(combined)


class hard_sample_aware_network(nn.Module):
    """
    DUET network

    Architecture:
        X [N,D] -> Attribute encoder -> Z_attr [N,H]  -+-> Fusion -> Z_final
        A [N,N] -> Structure encoder -> Z_stru [N,H]  -+
                                        |
                                   Classification head -> logits
    """
    def __init__(self, input_dim, hidden_dim, act, n_num, num_classes):
        super().__init__()

        # Get activation function
        act_fn = ACTIVATIONS.get(act, nn.Identity)()

        # Attribute encoder: X [N, input_dim] -> [N, hidden_dim]
        self.attr_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_fn
        )

        # Structure encoder: A [N, n_num] -> [N, hidden_dim]
        self.stru_encoder = nn.Sequential(
            nn.Linear(n_num, hidden_dim),
            act_fn
        )

        # Decoder: reconstruct original features
        self.decoder = nn.Linear(hidden_dim, input_dim)

        # Symmetric cross-attention fusion
        self.fusion = SymmetricCrossAttentionFusion(hidden_dim)

        # Classification heads
        self.head_attr = nn.Linear(hidden_dim, num_classes)
        self.head_stru = nn.Linear(hidden_dim, num_classes)
        self.head_final = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, A):
        """
        Args:
            x: Node features [N, D]
            A: Adjacency matrix [N, N]
        Returns:
            Z_attr, Z_stru: L2-normalized representations
            logits_attr, logits_stru: Classification logits
            Z_final: Fused representation
            X_hat: Reconstructed features
        """
        # Encode
        z_attr_raw = self.attr_encoder(x)
        z_stru_raw = self.stru_encoder(A)

        # Reconstruct
        x_hat = self.decoder(z_attr_raw)

        # L2 normalize
        z_attr = F.normalize(z_attr_raw, p=2, dim=1)
        z_stru = F.normalize(z_stru_raw, p=2, dim=1)

        # Fuse
        z_fused = self.fusion(z_attr, z_stru)
        z_final = F.normalize(self.head_final(z_fused), p=2, dim=1)

        # Classify
        logits_attr = self.head_attr(z_attr)
        logits_stru = self.head_stru(z_stru)

        return z_attr, z_stru, logits_attr, logits_stru, z_fused, x_hat
