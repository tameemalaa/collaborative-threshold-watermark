"""
Simulation of Fully Trustless GJKR-based Key Generation

Protocol demonstrates:
1. Commitment-based sampling (watermarks are additive shares)
2. MPC normalization (norm only revealed)
3. GJKR distributed Shamir sharing

Key property: Normalized watermark K is NEVER revealed to any party!
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trustless_keygen_distributed import (
    trustless_keygen_with_gjkr,
    reconstruct_from_gjkr_shares,
    verify_commitment
)
from src.ResNet import ResNet18


def main():
    # Configuration
    K = 4  # Number of parties
    threshold = 3  # t-of-n threshold (majority)
    seed = 1

    # Initialize model on GPU 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ResNet18().to(device)

    print("="*70)
    print("FULLY TRUSTLESS KEY GENERATION WITH GJKR")
    print("="*70)
    print(f"Configuration: {K} parties, {threshold}/{K} threshold, device: {device}")
    print("="*70)

    final_shares, original_normalized, timing_stats = trustless_keygen_with_gjkr(
        model, K, threshold, seed, device
    )

    print(f"✓ Protocol completed successfully!\n")

    print(f"{'='*70}")
    print(f"PERFORMANCE METRICS")
    print(f"{'='*70}")

    # Computation Time
    print(f"\nCOMPUTATION TIME (measured):")
    print(f"  Total computation time:   {timing_stats['total_time']:.3f}s")
    print(f"\n  Per-Phase Breakdown:")
    print(f"    Phase 1 (Sampling):       {timing_stats['sampling_time']:.3f}s")
    print(f"    Phase 2 (Additive Share): {timing_stats['sharing_time']:.3f}s")
    print(f"    Phase 3 (MPC Normalize):  {timing_stats['mpc_time']:.3f}s")
    print(f"    Phase 4 (GJKR Shamir):    {timing_stats['gjkr_time']:.3f}s")

    # Communication Cost
    print(f"\nCOMMUNICATION COST (bytes transferred):")
    print(f"  Total communication:      {timing_stats['total_comm_bytes'] / 1024 / 1024:.2f} MB")
    print(f"  Total rounds:             {timing_stats['total_rounds']}")
    print(f"\n  Per-Phase Breakdown:")
    print(f"    Phase 3 (MPC):            {timing_stats['mpc_comm_bytes'] / 1024 / 1024:.2f} MB ({timing_stats['mpc_rounds']} rounds)")
    print(f"    Phase 4 (GJKR):           {timing_stats['gjkr_comm_bytes'] / 1024 / 1024:.2f} MB ({timing_stats['gjkr_rounds']} round)")
    print(f"{'='*70}")

    # Verification: Reconstruct with threshold shares
    print(f"\n{'='*70}")
    print(f"VERIFICATION (t={threshold} shares)")
    print(f"{'='*70}")

    reconstructed = reconstruct_from_gjkr_shares(final_shares, threshold)

    # Verify normalization
    print(f"\nNormalization Check (each layer should have norm ≈ 1.0):")
    max_error = 0.0
    norm_errors = []
    for name, tensor in reconstructed.items():
        if tensor is not None:
            norm = tensor.norm().item()
            error = abs(norm - 1.0)
            max_error = max(max_error, error)
            norm_errors.append(error)
            if error > 0.001:  # Print layers with significant error
                print(f"  {name}: norm={norm:.6f}, error={error:.6f}")

    print(f"\n  Max norm error: {max_error:.8f}")
    print(f"  Mean norm error: {sum(norm_errors)/len(norm_errors):.8f}")

    # Calculate cosine similarity
    print(f"\nCosine Similarity (original vs reconstructed):")
    print(f"  (1.0 = perfect reconstruction, <0.99 = potential issue)")
    min_cosine_sim = 1.0
    cosine_sims = []
    bad_layers = []

    for name in original_normalized.keys():
        if original_normalized[name] is not None and reconstructed[name] is not None:
            original = original_normalized[name].cpu().flatten()
            recon = reconstructed[name].cpu().flatten()

            dot_product = (original * recon).sum().item()
            norm_original = original.norm().item()
            norm_recon = recon.norm().item()
            cosine_sim = dot_product / (norm_original * norm_recon)

            cosine_sims.append(cosine_sim)
            min_cosine_sim = min(min_cosine_sim, cosine_sim)

            if cosine_sim < 0.99:
                bad_layers.append((name, cosine_sim))

    avg_cosine_sim = sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0
    print(f"  Average: {avg_cosine_sim:.10f}")
    print(f"  Minimum: {min_cosine_sim:.10f}")
    print(f"  Maximum: {max(cosine_sims):.10f}")

    if bad_layers:
        print(f"\n  WARNING: Layers with cosine similarity < 0.99:")
        for name, sim in bad_layers:
            print(f"    {name}: {sim:.10f}")
    else:
        print(f"\n  ✓ All layers have cosine similarity ≥ 0.99")

    # Additional verification: check if reconstruction matches original
    print(f"\nDirect Comparison:")
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    for name in original_normalized.keys():
        if original_normalized[name] is not None and reconstructed[name] is not None:
            original = original_normalized[name].cpu()
            recon = reconstructed[name].cpu()

            abs_diff = (original - recon).abs().max().item()
            rel_diff = ((original - recon).abs() / (original.abs() + 1e-10)).max().item()

            max_abs_diff = max(max_abs_diff, abs_diff)
            max_rel_diff = max(max_rel_diff, rel_diff)

    print(f"  Max absolute difference: {max_abs_diff:.10f}")
    print(f"  Max relative difference: {max_rel_diff:.10f}")

    # Calculate statistics
    total_params = sum(t.numel() if t is not None else 0
                      for t in reconstructed.values())

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Parameters: {total_params:,}")
    print(f"  Parties: {K}, Threshold: {threshold}/{K} ({threshold/K*100:.0f}%)")
    print(f"  ✓ NO TRUSTED DEALER (fully distributed GJKR)")
    print(f"  ✓ K NEVER REVEALED (stays secret-shared)")


if __name__ == "__main__":
    main()
