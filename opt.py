"""
Command line arguments
"""
import argparse

parser = argparse.ArgumentParser()

# === Dataset ===
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--dataset', type=str, default='cora')

# === Preprocessing ===
parser.add_argument('--n_input', type=int, default=-1, help='PCA dimension, -1 means no PCA')
parser.add_argument('--t', type=int, default=2, help='Laplacian filtering times')

# === Network ===
parser.add_argument('--dims', type=int, default=1500, help='Hidden dimension')
parser.add_argument('--activate', type=str, default='ident', help='Activation function')

# === Training ===
parser.add_argument('--runs', type=int, default=10, help='Number of runs')
parser.add_argument('--epochs', type=int, default=600, help='Training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')

# === Dynamic parameters ===
parser.add_argument('--temp_start', type=float, default=0.8, help='Start temperature')
parser.add_argument('--temp_end', type=float, default=0.3, help='End temperature')
parser.add_argument('--sep_rate_start', type=float, default=0.05, help='Start separation rate')
parser.add_argument('--sep_rate_end', type=float, default=0.4, help='End separation rate')

# === Progressive weights ===
parser.add_argument('--guidance_weight_start', type=float, default=0.05, help='Start guidance weight')
parser.add_argument('--guidance_weight_end', type=float, default=0.5, help='End guidance weight')

# === Results ===
parser.add_argument('--acc', type=float, default=0)
parser.add_argument('--nmi', type=float, default=0)
parser.add_argument('--ari', type=float, default=0)
parser.add_argument('--f1', type=float, default=0)

args, _ = parser.parse_known_args()
