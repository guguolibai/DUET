"""
Training script

Core logic:
1. Contrastive learning (representation-level + logits-level)
2. Gamma-score entrusted teaching (trustworthy sample pseudo-labels)
3. Progressive weight scheduling
"""
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch import optim

from setup import setup_args
from model import hard_sample_aware_network
from utils import *


def compute_losses(model, x, A, args, subgraphs, epoch):
    """
    Compute all losses

    Returns:
        loss_total: total loss
        n_guided: number of samples in entrusted teaching
    """
    # Forward pass
    z_attr, z_stru, l_a, l_s, z_final, x_hat = model(x, A)

    # Training progress [0, 1]
    progress = epoch / args.epochs

    # === Dynamic parameters ===
    temp = args.temp_start - (args.temp_start - args.temp_end) * progress
    rate = args.sep_rate_start + (args.sep_rate_end - args.sep_rate_start) * progress
    guidance_w = args.guidance_weight_start + (args.guidance_weight_end - args.guidance_weight_start) * progress

    # === Loss 1: Contrastive learning ===
    # Representation-level: attr-stru + fused-attr + fused-stru
    loss_cl = contrastive_loss(z_attr, z_stru, temp)
    loss_cl += contrastive_loss(l_a, l_s, temp)

    # === Loss 2: Entrusted teaching ===
    # Compute Gamma scores, separate trustworthy/uncertain samples
    gamma_a = calculate_gamma_score(l_a, l_s, subgraphs)
    gamma_s = calculate_gamma_score(l_s, l_a, subgraphs)

    mask_I_a, mask_I_s = get_sample_mask(gamma_a, rate), get_sample_mask(gamma_s, rate)

    # Pseudo-labels
    pseudo_a, pseudo_s = l_a.argmax(dim=1).detach(), l_s.argmax(dim=1).detach()

    ce = nn.CrossEntropyLoss(reduction='none')

    # Trustworthy samples: cross-entropy
    loss_I_a = ce(l_a[mask_I_s], pseudo_s[mask_I_s]).mean() if mask_I_s.sum() > 0 else 0
    loss_I_s = ce(l_s[mask_I_a], pseudo_a[mask_I_a]).mean() if mask_I_a.sum() > 0 else 0

    loss_guidance = loss_I_a + loss_I_s

    # === Total loss ===
    loss_total = loss_cl + guidance_w * loss_guidance

    return loss_total, mask_I_a.sum().item()


def train(dataset_name):
    """Main training function"""
    args = setup_args(dataset_name)
    args.device = "cuda:0"

    # record results
    file = open(file_name, "a+")
    print(args.dataset, file=file)
    print("ACC,   NMI,   ARI,   F1", file=file)
    file.close()
    acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []

    for seed in range(args.runs):
        setup_seed(seed)
        args.acc = args.nmi = args.ari = args.f1 = 0
        best_Z, best_P = None, None

        # Load data
        X, y, A, n_nodes, n_classes = load_graph_data(dataset_name, args.n_input)
        X_filtered = laplacian_filtering(A, X, args.t)
        # edge_index = adj_to_edge_index(A)
        edge_index = A.nonzero(as_tuple=False).t()
        subgraphs = compute_khop_weight_matrix(edge_index, n_nodes)

        # Build model
        model = hard_sample_aware_network(
            input_dim=X.shape[1],
            hidden_dim=args.dims,
            act=args.activate,
            n_num=n_nodes,
            num_classes=n_classes
        ).to(args.device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        A, X_filtered = A.to(args.device), X_filtered.to(args.device)

        for epoch in tqdm(range(args.epochs), desc=f"Seed {seed}"):
            model.train()

            loss, n_guided = compute_losses(model, X_filtered, A, args, subgraphs, epoch)

            loss.backward()
            optimizer.step()

            # Evaluate every epoch
            if (epoch + 1) % 1 == 0:
                model.eval()
                z_attr, z_stru, l_attr, l_stru, z_final, _ = model(X_filtered, A)
                acc, nmi, ari, f1, P, center = phi(z_final, y, n_classes)

                if acc >= args.acc:
                    args.acc, args.nmi, args.ari, args.f1 = acc, nmi, ari, f1
                    best_Z = z_final
                    best_P = P

        print("Training complete")

        # record results
        file = open(file_name, "a+")
        print("{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(args.acc, args.nmi, args.ari, args.f1), file=file)
        file.close()
        acc_list.append(args.acc)
        nmi_list.append(args.nmi)
        ari_list.append(args.ari)
        f1_list.append(args.f1)

    # record results
    acc_list, nmi_list, ari_list, f1_list = map(lambda x: np.array(x), (acc_list, nmi_list, ari_list, f1_list))
    file = open(file_name, "a+")
    print("{:.2f}, {:.2f}".format(acc_list.mean(), acc_list.std()), file=file)
    print("{:.2f}, {:.2f}".format(nmi_list.mean(), nmi_list.std()), file=file)
    print("{:.2f}, {:.2f}".format(ari_list.mean(), ari_list.std()), file=file)
    print("{:.2f}, {:.2f}".format(f1_list.mean(), f1_list.std()), file=file)
    file.close()


if __name__ == '__main__':
    file_name = "result.csv"

    for dataset in ["cite"]:
        train(dataset)
