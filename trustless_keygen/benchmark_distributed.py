"""
Benchmark Trustless Key Generation with Varying Number of Clients

This script runs experiments with different numbers of parties and generates:
1. CSV files with detailed metrics
2. Publication-quality plots for papers

Network speed assumption: 1 Gbps (125 MB/s) for communication time estimation
"""

import torch
import sys
import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trustless_keygen_distributed import trustless_keygen_with_gjkr, reconstruct_from_gjkr_shares
from src.ResNet import ResNet18

# Network bandwidth for communication time estimation
NETWORK_BANDWIDTH_MBPS = 1000  # 1 Gbps
NETWORK_BANDWIDTH_MBps = NETWORK_BANDWIDTH_MBPS / 8  # 125 MB/s


def run_experiment(n_parties, threshold_ratio=0.5, seed=42, device='cuda:0'):
    """
    Run single experiment with given number of parties.

    Args:
        n_parties: Number of parties
        threshold_ratio: Threshold as ratio of n_parties (0.5 = majority)
        seed: Random seed
        device: Device to run on

    Returns:
        Dictionary with all metrics
    """
    threshold = max(2, int(n_parties * threshold_ratio))

    print(f"\n{'='*70}")
    print(f"Running experiment: {n_parties} parties, {threshold}/{n_parties} threshold")
    print(f"{'='*70}")

    # Initialize model
    model = ResNet18().to(device)

    # Run protocol
    final_shares, original_normalized, timing_stats = trustless_keygen_with_gjkr(
        model, n_parties, threshold, seed, device
    )

    # Calculate communication time based on bandwidth
    comm_time_mpc = timing_stats['mpc_comm_bytes'] / (NETWORK_BANDWIDTH_MBps * 1024 * 1024)
    comm_time_gjkr = timing_stats['gjkr_comm_bytes'] / (NETWORK_BANDWIDTH_MBps * 1024 * 1024)
    total_comm_time = comm_time_mpc + comm_time_gjkr

    # Calculate total parameters
    total_params = sum(t.numel() if t is not None else 0
                      for t in original_normalized.values())

    # Reconstruct and verify
    print(f"  Reconstructing from {threshold} shares...")
    reconstructed = reconstruct_from_gjkr_shares(final_shares, threshold)

    # Verify normalization
    max_norm_error = 0.0
    for name, tensor in reconstructed.items():
        if tensor is not None:
            norm = tensor.norm().item()
            error = abs(norm - 1.0)
            max_norm_error = max(max_norm_error, error)

    # Calculate cosine similarity
    cosine_sims = []
    for name in original_normalized.keys():
        if original_normalized[name] is not None and reconstructed[name] is not None:
            original = original_normalized[name].cpu().flatten()
            recon = reconstructed[name].cpu().flatten()

            dot_product = (original * recon).sum().item()
            norm_original = original.norm().item()
            norm_recon = recon.norm().item()
            cosine_sim = dot_product / (norm_original * norm_recon)

            cosine_sims.append(cosine_sim)

    avg_cosine_sim = sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0
    min_cosine_sim = min(cosine_sims) if cosine_sims else 0

    # Calculate per-client metrics
    # In GJKR, each party sends to (n-1) other parties and receives from (n-1) parties
    # So per-party communication is: total_comm / n_parties
    comm_mb_per_client = (timing_stats['total_comm_bytes'] / (1024 * 1024)) / n_parties
    comm_time_per_client = total_comm_time / n_parties

    # Computation time per client
    # NOTE: Simulation runs sequentially, so divide by n_parties for parallel equivalent
    # In real parallel deployment, each client would do 1/n of the total work
    comp_time_per_client = timing_stats['total_time'] / n_parties

    # Total time per client
    total_time_per_client = comp_time_per_client + comm_time_per_client

    # Compile results
    results = {
        'n_parties': n_parties,
        'threshold': threshold,
        'threshold_ratio': threshold_ratio,
        'total_params': total_params,

        # Per-client computation time (seconds)
        'comp_time_per_client': comp_time_per_client,

        # Per-client communication MB
        'comm_mb_per_client': comm_mb_per_client,

        # Per-client communication time (seconds) at 1 Gbps
        'comm_time_per_client': comm_time_per_client,

        # Per-client total time (computation + communication)
        'total_time_per_client': total_time_per_client,

        # Rounds
        'comm_rounds_total': timing_stats['total_rounds'],

        # Network bandwidth used
        'network_bandwidth_gbps': NETWORK_BANDWIDTH_MBPS / 1000,

        # Verification metrics
        'max_norm_error': max_norm_error,
        'avg_cosine_similarity': avg_cosine_sim,
        'min_cosine_similarity': min_cosine_sim,
    }

    return results


def run_all_experiments(party_counts, threshold_ratio=0.5, output_dir='benchmark_results'):
    """
    Run experiments for all party counts.

    Args:
        party_counts: List of party counts to test
        threshold_ratio: Threshold ratio (default 0.5 = majority)
        output_dir: Directory to save results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Network bandwidth: {NETWORK_BANDWIDTH_MBPS} Mbps ({NETWORK_BANDWIDTH_MBps} MB/s)")

    all_results = []

    for n_parties in party_counts:
        try:
            results = run_experiment(n_parties, threshold_ratio, device=device)
            all_results.append(results)

            print(f"\n✓ Completed: {n_parties} parties")
            print(f"  Per-client time: {results['total_time_per_client']:.3f}s (comp: {results['comp_time_per_client']:.3f}s + comm: {results['comm_time_per_client']:.3f}s)")
            print(f"  Per-client communication: {results['comm_mb_per_client']:.2f} MB")
            print(f"  Cosine similarity: {results['avg_cosine_similarity']:.6f}, Norm error: {results['max_norm_error']:.6f}")

        except Exception as e:
            print(f"\n✗ Failed: {n_parties} parties - {e}")
            import traceback
            traceback.print_exc()

    return all_results


def save_to_csv(results, filename='benchmark_results/results.csv'):
    """Save results to CSV file."""
    if not results:
        print("No results to save")
        return

    fieldnames = list(results[0].keys())

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ Saved results to {filename}")


def create_plots(results, output_dir='benchmark_results'):
    """Create publication-quality plots."""
    if not results:
        print("No results to plot")
        return

    # Set publication style
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
        'lines.markersize': 8
    })

    # Extract data
    n_parties = [r['n_parties'] for r in results]

    # Plot 1: Per-Client Time (Computation + Communication)
    fig, ax = plt.subplots(figsize=(10, 6))

    total_time_per_client = [r['total_time_per_client'] for r in results]
    comp_per_client = [r['comp_time_per_client'] for r in results]
    comm_time_per_client = [r['comm_time_per_client'] for r in results]

    ax.plot(n_parties, total_time_per_client, 'o-', label='Total Time per Client', linewidth=2.5, markersize=10, color='purple')
    ax.plot(n_parties, comp_per_client, 's--', label='Computation Time per Client', alpha=0.7)
    ax.plot(n_parties, comm_time_per_client, '^--', label='Communication Time per Client', alpha=0.7)

    ax.set_xlabel('Number of Parties')
    ax.set_ylabel('Time per Client (seconds)')
    ax.set_title(f'Per-Client Protocol Time vs Number of Parties (@ {NETWORK_BANDWIDTH_MBPS} Mbps)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/per_client_time.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/per_client_time.pdf', bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/per_client_time.png")

    # Plot 2: Per-Client Communication Cost (MB) vs Number of Parties
    fig, ax = plt.subplots(figsize=(10, 6))

    comm_mb_per_client = [r['comm_mb_per_client'] for r in results]

    ax.plot(n_parties, comm_mb_per_client, 'o-', linewidth=2.5, markersize=10, color='green')

    ax.set_xlabel('Number of Parties')
    ax.set_ylabel('Communication Cost per Client (MB)')
    ax.set_title('Per-Client Communication Cost vs Number of Parties')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/per_client_communication.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/per_client_communication.pdf', bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/per_client_communication.png")

    # Plot 3: Cosine Similarity
    fig, ax = plt.subplots(figsize=(10, 6))

    avg_cosine = [r['avg_cosine_similarity'] for r in results]
    min_cosine = [r['min_cosine_similarity'] for r in results]

    ax.plot(n_parties, avg_cosine, 'o-', label='Average Cosine Similarity', linewidth=2.5, markersize=10, color='blue')
    ax.plot(n_parties, min_cosine, 's--', label='Minimum Cosine Similarity', linewidth=2, markersize=8, alpha=0.7, color='cyan')

    ax.set_xlabel('Number of Parties')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Watermark Reconstruction Accuracy vs Number of Parties')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.99, 1.001])  # Focus on near-perfect similarity

    plt.tight_layout()
    plt.savefig(f'{output_dir}/cosine_similarity.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/cosine_similarity.pdf', bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/cosine_similarity.png")

    # Plot 4: Normalization Error
    fig, ax = plt.subplots(figsize=(10, 6))

    norm_error = [r['max_norm_error'] for r in results]

    ax.plot(n_parties, norm_error, 'o-', linewidth=2.5, markersize=10, color='red')

    ax.set_xlabel('Number of Parties')
    ax.set_ylabel('Max Norm Error')
    ax.set_title('Normalization Error vs Number of Parties')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for small errors

    plt.tight_layout()
    plt.savefig(f'{output_dir}/norm_error.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/norm_error.pdf', bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/norm_error.png")

    plt.close('all')


def main():
    """Main benchmarking function."""
    print("="*70)
    print("TRUSTLESS KEY GENERATION BENCHMARK")
    print("="*70)
    print(f"Network bandwidth: {NETWORK_BANDWIDTH_MBPS} Mbps ({NETWORK_BANDWIDTH_MBps} MB/s)")
    print("="*70)

    # Configuration
    party_counts = [4, 8, 16]  # Different numbers of parties
    threshold_ratio = 0.5  # Majority threshold
    output_dir = 'benchmark_results'

    print(f"\nTesting party counts: {party_counts}")
    print(f"Threshold ratio: {threshold_ratio} (majority)")
    print(f"Output directory: {output_dir}\n")

    # Run experiments
    results = run_all_experiments(party_counts, threshold_ratio, output_dir)

    # Save to CSV
    csv_file = f'{output_dir}/results.csv'
    save_to_csv(results, csv_file)

    # Create plots
    print(f"\nGenerating plots...")
    create_plots(results, output_dir)

    # Summary
    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {csv_file}")
    print(f"Plots saved to: {output_dir}/")
    print(f"  - per_client_time.png/pdf (computation + communication per client)")
    print(f"  - per_client_communication.png/pdf")
    print(f"  - cosine_similarity.png/pdf")
    print(f"  - norm_error.png/pdf")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
