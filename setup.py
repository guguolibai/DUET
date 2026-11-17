"""
Hyperparameter configuration for different datasets
"""
from opt import args

# Dataset-specific configurations
CONFIGS = {}


def setup_args(dataset_name):
    """Load dataset-specific hyperparameters into args"""
    args.dataset = dataset_name
    args.acc = args.nmi = args.ari = args.f1 = 0

    # Load config (use default from opt.py if dataset not found)
    cfg = CONFIGS.get(dataset_name, {})
    for key, value in cfg.items():
        setattr(args, key, value)

    print("=" * 40)
    print(f"Dataset: {dataset_name}")
    print(f"Hidden: {args.dims}, LR: {args.lr}, Epochs: {args.epochs}")
    print(f"Filter: {args.t}, Activate: {args.activate}")
    print("=" * 40)

    return args
