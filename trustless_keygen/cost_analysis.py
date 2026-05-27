"""
Theoretical Cost Analysis for Trustless Key Generation

Calculates communication and computation costs without executing the protocol.
Useful for quick cost estimation and scalability analysis.
"""

import torch
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ResNet import ResNet18


def count_model_parameters(model: torch.nn.Module) -> dict:
    """Count trainable parameters in the model."""
    total_params = 0
    layer_params = {}
    num_layers = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            params = param.numel()
            layer_params[name] = params
            total_params += params
            num_layers += 1

    return {
        'total_params': total_params,
        'num_layers': num_layers,
        'layer_params': layer_params
    }


def calculate_phase1_costs(n_parties: int, total_params: int) -> dict:
    """
    Phase 1: Sampling with commitments

    Each party:
    - Samples random vector (no communication)
    - Creates commitment hash (no communication)

    Communication: commitments exchanged (optional, depends on protocol variant)
    """
    # Sampling computation: generate random integers
    sampling_ops = n_parties * total_params

    # Hash computation (SHA256): ~64 bytes per hash, one hash per layer
    # Simplified: assume negligible compared to later phases

    # Communication: broadcast commitments (32 bytes per commitment hash)
    # In current implementation, commitments stored locally (0 bytes)
    comm_bytes = 0

    return {
        'sampling_ops': sampling_ops,
        'comm_bytes': comm_bytes,
        'comm_rounds': 0
    }


def calculate_phase2_costs(n_parties: int, total_params: int) -> dict:
    """
    Phase 2: MPC Normalization

    Protocol:
    1. Each party encrypts their share (n encryptions)
    2. MPC sum: n-1 additions in encrypted domain
    3. MPC square and sum: total_params multiplications + additions
    4. Reveal norm (1 scalar)
    5. Each party divides locally by norm

    Communication:
    - Share encrypted values: n parties × total_params × 8 bytes
    - Encrypted operations: multiplication protocol overhead
    - Reveal: 1 scalar (8 bytes)
    """
    # Encryption operations
    encryption_ops = n_parties * total_params

    # MPC additions for sum
    mpc_additions = (n_parties - 1) * total_params

    # MPC multiplications for norm (element-wise square)
    mpc_multiplications = total_params

    # Total MPC operations
    mpc_ops = mpc_additions + mpc_multiplications

    # Communication:
    # - Initial sharing: each party sends to all others
    # - Estimated 3 rounds (share, multiply, reveal)
    bytes_per_param = 8  # float64
    comm_bytes = n_parties * total_params * bytes_per_param * 3

    return {
        'encryption_ops': encryption_ops,
        'mpc_ops': mpc_ops,
        'mpc_additions': mpc_additions,
        'mpc_multiplications': mpc_multiplications,
        'comm_bytes': comm_bytes,
        'comm_rounds': 3
    }


def calculate_phase3_costs(n_parties: int, threshold: int, total_params: int) -> dict:
    """
    Phase 3: GJKR Shamir Sharing

    Each party:
    1. Creates polynomial of degree (threshold-1)
    2. Evaluates polynomial at n points
    3. Sends n-1 shares to other parties
    4. Receives n-1 shares from other parties
    5. Sums received shares

    Communication:
    - Each party sends (n-1) shares of size total_params
    - Total: n × (n-1) × total_params × 8 bytes
    """
    # Polynomial creation: threshold coefficients per parameter
    poly_creation_ops = n_parties * total_params * threshold

    # Polynomial evaluation: Horner's method, threshold-1 muls + threshold-1 adds per point
    eval_ops_per_point = 2 * (threshold - 1)
    poly_eval_ops = n_parties * total_params * n_parties * eval_ops_per_point

    # Share aggregation: each party sums n shares
    aggregation_ops = n_parties * total_params * n_parties

    total_ops = poly_creation_ops + poly_eval_ops + aggregation_ops

    # Communication: n parties send to (n-1) others
    bytes_per_param = 8  # float64
    comm_bytes = n_parties * (n_parties - 1) * total_params * bytes_per_param

    return {
        'poly_creation_ops': poly_creation_ops,
        'poly_eval_ops': poly_eval_ops,
        'aggregation_ops': aggregation_ops,
        'total_ops': total_ops,
        'comm_bytes': comm_bytes,
        'comm_rounds': 1
    }


def analyze_costs(n_parties: int, threshold: int, model_name: str = "ResNet18"):
    """Run complete cost analysis."""
    print(f"\n{'='*70}")
    print(f"TRUSTLESS KEYGEN COST ANALYSIS")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Parties: {n_parties}")
    print(f"Threshold: {threshold}/{n_parties} ({threshold/n_parties*100:.0f}%)")
    print(f"{'='*70}\n")

    # Count model parameters
    if model_name == "ResNet18":
        model = ResNet18()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    param_info = count_model_parameters(model)
    total_params = param_info['total_params']
    num_layers = param_info['num_layers']

    print(f"MODEL PARAMETERS")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Size (float64): {total_params * 8 / 1024 / 1024:.2f} MB\n")

    # Phase 1: Sampling
    phase1 = calculate_phase1_costs(n_parties, total_params)
    print(f"PHASE 1: SAMPLING WITH COMMITMENTS")
    print(f"  Sampling operations: {phase1['sampling_ops']:,}")
    print(f"  Communication: {phase1['comm_bytes'] / 1024 / 1024:.2f} MB")
    print(f"  Rounds: {phase1['comm_rounds']}\n")

    # Phase 2: MPC Normalization
    phase2 = calculate_phase2_costs(n_parties, total_params)
    print(f"PHASE 2: MPC NORMALIZATION")
    print(f"  Encryption operations: {phase2['encryption_ops']:,}")
    print(f"  MPC operations: {phase2['mpc_ops']:,}")
    print(f"    - Additions: {phase2['mpc_additions']:,}")
    print(f"    - Multiplications: {phase2['mpc_multiplications']:,}")
    print(f"  Communication: {phase2['comm_bytes'] / 1024 / 1024:.2f} MB")
    print(f"  Rounds: {phase2['comm_rounds']}\n")

    # Phase 3: GJKR Shamir
    phase3 = calculate_phase3_costs(n_parties, threshold, total_params)
    print(f"PHASE 3: GJKR SHAMIR SHARING")
    print(f"  Polynomial creation: {phase3['poly_creation_ops']:,} ops")
    print(f"  Polynomial evaluation: {phase3['poly_eval_ops']:,} ops")
    print(f"  Share aggregation: {phase3['aggregation_ops']:,} ops")
    print(f"  Total operations: {phase3['total_ops']:,}")
    print(f"  Communication: {phase3['comm_bytes'] / 1024 / 1024:.2f} MB")
    print(f"  Rounds: {phase3['comm_rounds']}\n")

    # Total costs
    total_ops = (phase1['sampling_ops'] +
                 phase2['mpc_ops'] +
                 phase3['total_ops'])
    total_comm = (phase1['comm_bytes'] +
                  phase2['comm_bytes'] +
                  phase3['comm_bytes'])
    total_rounds = (phase1['comm_rounds'] +
                    phase2['comm_rounds'] +
                    phase3['comm_rounds'])

    print(f"{'='*70}")
    print(f"TOTAL COSTS")
    print(f"{'='*70}")
    print(f"  Total operations: {total_ops:,}")
    print(f"  Total communication: {total_comm / 1024 / 1024:.2f} MB")
    print(f"  Total rounds: {total_rounds}")

    # Per-client costs
    print(f"\nPER-CLIENT COSTS")
    print(f"  Operations per client: {total_ops / n_parties:,.0f}")
    print(f"  Communication per client: {total_comm / n_parties / 1024 / 1024:.2f} MB")

    # Bandwidth requirements (assuming 1 Gbps network)
    network_gbps = 1.0
    network_mbps = network_gbps * 1000
    network_MBps = network_mbps / 8

    total_comm_time = (total_comm / 1024 / 1024) / network_MBps
    comm_time_per_client = total_comm_time / n_parties

    print(f"\nNETWORK TIME (@ {network_gbps} Gbps)")
    print(f"  Total communication time: {total_comm_time:.2f}s")
    print(f"  Communication time per client: {comm_time_per_client:.2f}s")

    print(f"\n{'='*70}\n")

    return {
        'n_parties': n_parties,
        'threshold': threshold,
        'total_params': total_params,
        'num_layers': num_layers,
        'phase1': phase1,
        'phase2': phase2,
        'phase3': phase3,
        'total_ops': total_ops,
        'total_comm_bytes': total_comm,
        'total_rounds': total_rounds
    }


def plot_communication_time(results: list, network_gbps: float = 1.0, output_file: str = 'communication_time_per_client.png'):
    """Generate plot of number of clients vs communication time per client."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.figsize': (10, 6),
        'lines.linewidth': 2,
        'lines.markersize': 10
    })

    network_MBps = network_gbps * 1000 / 8

    n_parties_list = [r['n_parties'] for r in results]
    comm_time_per_client = []

    for r in results:
        total_comm_mb = r['total_comm_bytes'] / 1024 / 1024
        total_time = total_comm_mb / network_MBps
        time_per_client = total_time / r['n_parties']
        comm_time_per_client.append(time_per_client)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_parties_list, comm_time_per_client, 'o-', linewidth=2.5, markersize=10, color='blue')

    ax.set_xlabel('Number of Clients')
    ax.set_ylabel('Communication Time per Client (seconds)')
    ax.set_title(f'Communication Time per Client vs Number of Clients (@ {network_gbps} Gbps)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nPlot saved: {output_file}")
    plt.close()


def compare_scalability(party_counts: list, threshold_ratio: float = 0.5, generate_plot: bool = True):
    """Compare costs across different numbers of parties."""
    print(f"\n{'='*70}")
    print(f"SCALABILITY ANALYSIS")
    print(f"{'='*70}\n")

    results = []
    for n_parties in party_counts:
        threshold = max(2, int(n_parties * threshold_ratio))
        result = analyze_costs(n_parties, threshold)
        results.append(result)

    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Parties':<10} {'Threshold':<12} {'Comm (MB)':<15} {'Ops (M)':<15}")
    print(f"{'-'*70}")

    for r in results:
        ops_millions = r['total_ops'] / 1e6
        comm_mb = r['total_comm_bytes'] / 1024 / 1024
        print(f"{r['n_parties']:<10} {r['threshold']:<12} {comm_mb:<15.2f} {ops_millions:<15.2f}")

    print(f"{'='*70}\n")

    # Generate plot
    if generate_plot:
        plot_communication_time(results)


def main():
    """Main cost analysis."""
    # Single configuration analysis
    analyze_costs(n_parties=16, threshold=9)

    # Scalability comparison
    party_counts = [4, 8, 16, 32, 64, 128]
    compare_scalability(party_counts)


if __name__ == "__main__":
    main()
