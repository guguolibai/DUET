import torch
import random
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
from munkres import Munkres
from kmeans_gpu import kmeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch_geometric.utils import degree, add_self_loops, k_hop_subgraph


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_graph_data(dataset_name, n_input=-1, show_details=False):
    path = f"dataset/{dataset_name}/{dataset_name}"
    feat = np.load(f"{path}_feat.npy", allow_pickle=True)
    label = np.load(f"{path}_label.npy", allow_pickle=True)
    adj = np.load(f"{path}_adj.npy", allow_pickle=True)

    # PCA (optional)
    if n_input > 0:
        feat = PCA(n_components=n_input).fit_transform(feat)

    if show_details:
        print(f"Dataset: {dataset_name}, Nodes: {feat.shape[0]}, "
              f"Features: {feat.shape[1]}, Classes: {len(np.unique(label))}")

    return feat, label, torch.tensor(adj).float(), feat.shape[0], len(np.unique(label))


def normalize_adj(adj, self_loop=True, symmetry=False):
    """
    normalize the adj matrix
    :param adj: input adj matrix
    :param self_loop: if add the self loop or not
    :param symmetry: symmetry normalize or not
    :return: the normalized adj matrix
    """
    # add the self_loop
    if self_loop:
        adj_tmp = adj + np.eye(adj.shape[0])
    else:
        adj_tmp = adj

    # calculate degree matrix and it's inverse matrix
    d = np.diag(adj_tmp.sum(0))
    d_inv = np.linalg.inv(d)

    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        sqrt_d_inv = np.sqrt(d_inv)
        norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), sqrt_d_inv)

    # non-symmetry normalize: D^{-1} A
    else:
        norm_adj = np.matmul(d_inv, adj_tmp)
    return norm_adj


def laplacian_filtering(A, X, t):
    A_tmp = A - torch.diag_embed(torch.diag(A))
    A_norm = normalize_adj(A_tmp, self_loop=True, symmetry=True)
    I = torch.eye(A.shape[0])
    L = I - A_norm
    for i in range(t):
        X = (I - 1 * L) @ X
    return X.float()


def adj_to_edge_index(adj):
    """Adjacency matrix -> edge_index [2, E]"""
    adj_np = adj.cpu().numpy() if torch.is_tensor(adj) else adj
    return torch.LongTensor(np.array(np.nonzero(adj_np)))


def compute_khop_weight_matrix(edge_index, num_nodes, k=2):
    """
    Precompute k-hop subgraph degree-weighted aggregation matrix.
    Returns sparse weight matrix W [N, N].
    """
    print(f"Precomputing {k}-hop weight matrix for {num_nodes} nodes...")

    rows, cols, vals = [], [], []

    for node in range(num_nodes):
        nodes, edges, _, _ = k_hop_subgraph(node, k, edge_index, relabel_nodes=True, num_nodes=num_nodes)
        nodes = nodes.tolist()

        if not nodes:
            rows.append(node)
            cols.append(node)
            vals.append(1.0)
            continue

        edges_with_self, _ = add_self_loops(edges, num_nodes=len(nodes))
        deg = degree(edges_with_self[0], len(nodes))
        weights = 1.0 / deg
        n = len(nodes)

        rows.extend([node] * n)
        cols.extend(nodes)
        vals.extend((weights / n).tolist())

    indices = torch.LongTensor([rows, cols])
    values = torch.FloatTensor(vals)
    W = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

    return W.coalesce()


def calculate_topological_uncertainty(logits, W):
    # W @ logits: [N, N] @ [N, C] -> [N, C]
    aggregated_logits = torch.sparse.mm(W.to(logits.device), logits)

    probs = F.softmax(aggregated_logits, dim=1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)
    return entropy


def calculate_gamma_score(logits_self, logits_other, W):
    # JS divergence
    p1 = F.softmax(logits_self, dim=-1) + 1e-6
    p2 = F.softmax(logits_other, dim=-1) + 1e-6
    m = (p1 + p2) / 2
    js = 0.5 * (F.kl_div(p1.log(), m, reduction="none").sum(1) +
                F.kl_div(p2.log(), m, reduction="none").sum(1))

    entropy = calculate_topological_uncertainty(logits_self, W)

    return entropy - js


def get_sample_mask(gamma, rate):
    k = int(rate * gamma.size(0))
    if k <= 0:
        return torch.zeros_like(gamma, dtype=torch.bool)

    threshold = torch.topk(gamma, k, largest=False)[0][-1]
    return gamma <= threshold


def contrastive_loss(z1, z2, temp=0.5):
    sim = z1 @ z2.T / temp
    pos = torch.diag(sim)
    return -(pos - torch.logsumexp(sim, dim=1)).mean()


def logits_contrastive_loss(l1, l2, temp=0.5):
    return contrastive_loss(l1, l2, temp)


def soft_contrastive_loss(l1, l2, temp=0.5):
    p1 = F.normalize(F.softmax(l1 / temp, dim=1), dim=1)
    p2 = F.normalize(F.softmax(l2 / temp, dim=1), dim=1)
    sim = p1 @ p2.T
    pos = torch.diag(sim)
    return -(pos - torch.logsumexp(sim, dim=1)).mean()


def cluster_acc(y_true, y_pred):
    y_true = y_true - y_true.min()
    classes = list(set(y_true))
    pred_classes = list(set(y_pred))

    cost = np.zeros((len(classes), len(pred_classes)), dtype=int)
    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(pred_classes):
            cost[i][j] = sum((y_true == c1) & (y_pred == c2))

    indices = Munkres().compute((-cost).tolist())

    mapping = {pred_classes[c]: classes[r] for r, c in indices}
    y_pred_aligned = np.array([mapping.get(p, p) for p in y_pred])

    acc = metrics.accuracy_score(y_true, y_pred_aligned)
    f1 = metrics.f1_score(y_true, y_pred_aligned, average='macro')
    return acc, f1


def eva(y_true, y_pred, show_details=True):
    """
    evaluate the clustering performance
    Args:
        y_true: the ground truth
        y_pred: the predicted label
        show_details: if print the details
    Returns: None
    """
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    ari = adjusted_rand_score(y_true, y_pred)
    if show_details:
        print(':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
              ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1

def phi(feature, true_labels, cluster_num):
    predict_labels, centers = kmeans(X=feature, num_clusters=cluster_num, distance="euclidean", device="cuda:0")
    acc, nmi, ari, f1 = eva(true_labels, predict_labels.numpy(), show_details=False)
    return 100 * acc, 100 * nmi, 100 * ari, 100 * f1, predict_labels.numpy(), centers
