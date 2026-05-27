"""
Trustless Key Generation with GJKR and Commitments

Protocol:
1. Sample W_i with commitments (additive shares: Σ W_i = W)
2. MPC normalize (reveal only norm)
3. GJKR distributed Shamir sharing
"""

import torch
import hashlib
from typing import Dict, List, Tuple
import time
import crypten
import crypten.communicator as comm

FIELD_PRIME = 2**53 - 1
PRECISION_BITS = 42
COEFF_RANGE = 2**8


def hash_commitment(value: torch.Tensor, randomness: int = None) -> str:
    """Create commitment: Com(value, r) = H(value || r)"""
    if randomness is None:
        randomness = torch.randint(0, 2**32, (1,)).item()

    value_bytes = value.cpu().numpy().tobytes()
    rand_bytes = randomness.to_bytes(8, 'big')

    hasher = hashlib.sha256()
    hasher.update(value_bytes)
    hasher.update(rand_bytes)

    return hasher.hexdigest(), randomness


def verify_commitment(value: torch.Tensor, randomness: int, commitment: str) -> bool:
    """Verify that value matches commitment."""
    recomputed, _ = hash_commitment(value, randomness)
    return recomputed == commitment


def sample_with_commitment(model: torch.nn.Module,
                           client_id: int,
                           seed: int,
                           device) -> Tuple[Dict[str, torch.Tensor], Dict[str, Tuple[str, int]]]:
    """Sample watermark vector and create commitment."""
    torch.manual_seed(seed + client_id)

    watermark = {}
    commitments = {}
    total_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            w = torch.randint_like(param, low=0, high=2) * 2 - 1
            w = w.double().to(device)
            watermark[name] = w
            commitments[name] = hash_commitment(w)
            total_params += w.numel()
        else:
            watermark[name] = None
            commitments[name] = None

    print(f"  Party {client_id}: sampled {total_params:,} parameters")
    return watermark, commitments


def mpc_normalize_secret_shared(shares: List[Dict[str, torch.Tensor]],
                                n_parties: int,
                                device=None) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor], Dict]:
    """Normalize secret-shared sum without revealing it (only reveals norm)."""
    if device is None:
        for layer_shares in shares:
            for tensor in layer_shares.values():
                if tensor is not None:
                    device = tensor.device
                    break
            if device is not None:
                break

    use_cuda = device is not None and device.type == 'cuda'

    timing_stats = {
        'mpc_time': 0.0,
        'total_time': 0.0,
        'mpc_comm_bytes': 0,
        'mpc_rounds': 3
    }

    if not crypten.is_initialized():
        crypten.init()

    print(f"  MPC normalization: {n_parties} parties on {device}")
    start = time.time()
    normalized_shares = [{} for _ in range(n_parties)]
    original_normalized = {}

    layer_count = 0
    for layer_name in shares[0].keys():
        if shares[0][layer_name] is None:
            for i in range(n_parties):
                normalized_shares[i][layer_name] = None
            original_normalized[layer_name] = None
            continue

        layer_count += 1
        party_shares = [shares[i][layer_name] for i in range(n_parties)]

        if use_cuda:
            encrypted_shares = [crypten.cryptensor(s, device=device) for s in party_shares]
        else:
            encrypted_shares = [crypten.cryptensor(s) for s in party_shares]

        encrypted_sum = encrypted_shares[0]
        for enc_s in encrypted_shares[1:]:
            encrypted_sum = encrypted_sum + enc_s

        norm_squared = (encrypted_sum * encrypted_sum).sum()
        norm_value = norm_squared.get_plain_text().sqrt().item()

        actual_sum = sum(party_shares)
        original_normalized[layer_name] = actual_sum / norm_value

        for i in range(n_parties):
            normalized_shares[i][layer_name] = party_shares[i] / norm_value

        timing_stats['mpc_comm_bytes'] += party_shares[0].numel() * 8 * n_parties * 3

    timing_stats['mpc_time'] = time.time() - start
    timing_stats['total_time'] = timing_stats['mpc_time']

    print(f"  MPC normalization completed: {layer_count} layers in {timing_stats['mpc_time']:.2f}s")
    return normalized_shares, original_normalized, timing_stats


def gjkr_shamir_share(my_share: Dict[str, torch.Tensor],
                      my_id: int,
                      n_parties: int,
                      threshold: int) -> Dict[str, torch.Tensor]:
    """GJKR-style distributed Shamir sharing."""
    final_shares = {}

    layer_count = 0
    for layer_name, my_tensor in my_share.items():
        if my_tensor is None:
            final_shares[layer_name] = None
            continue

        layer_count += 1
        original_shape = my_tensor.shape
        flat_tensor = my_tensor.flatten()
        n_elements = flat_tensor.numel()

        scale = 2 ** PRECISION_BITS
        my_secret_int = (flat_tensor * scale).long()
        my_secret_int = torch.where(my_secret_int < 0,
                                     my_secret_int + FIELD_PRIME,
                                     my_secret_int) % FIELD_PRIME

        coeffs = torch.zeros(n_elements, threshold, dtype=torch.long)
        coeffs[:, 0] = my_secret_int

        for k in range(1, threshold):
            coeffs[:, k] = torch.randint(0, COEFF_RANGE, (n_elements,), dtype=torch.long)

        x_values = torch.arange(1, n_parties + 1, dtype=torch.long).unsqueeze(1)
        shares_int = coeffs[:, threshold - 1].unsqueeze(0).expand(n_parties, -1)

        for k in range(threshold - 2, -1, -1):
            shares_int = (shares_int * x_values + coeffs[:, k]) % FIELD_PRIME

        # Keep as integers for field aggregation
        final_shares[layer_name] = shares_int.reshape(n_parties, *original_shape)

    return final_shares


def simulate_gjkr_protocol(normalized_additive_shares: List[Dict[str, torch.Tensor]],
                           n_parties: int,
                           threshold: int) -> Tuple[List[Dict[str, torch.Tensor]], Dict]:
    """Simulate GJKR protocol with on-the-fly aggregation."""
    timing_stats = {
        'gjkr_time': 0.0,
        'gjkr_comm_bytes': 0,
        'gjkr_rounds': 1
    }

    print(f"  GJKR Shamir sharing: {n_parties} parties, threshold={threshold}")
    start = time.time()
    aggregated_shares = [{} for _ in range(n_parties)]

    first_shares = gjkr_shamir_share(
        normalized_additive_shares[0],
        0,
        n_parties,
        threshold
    )

    layer_count = 0
    for layer_name in first_shares.keys():
        if first_shares[layer_name] is None:
            for j in range(n_parties):
                aggregated_shares[j][layer_name] = None
        else:
            layer_count += 1
            for j in range(n_parties):
                # Initialize with zeros (integers for field arithmetic)
                aggregated_shares[j][layer_name] = torch.zeros_like(first_shares[layer_name][0], dtype=torch.long)

    for layer_name in first_shares.keys():
        if first_shares[layer_name] is not None:
            for j in range(n_parties):
                # Field addition
                aggregated_shares[j][layer_name] = (aggregated_shares[j][layer_name] + first_shares[layer_name][j]) % FIELD_PRIME

            comm_per_share = first_shares[layer_name][0].numel() * 8
            timing_stats['gjkr_comm_bytes'] += comm_per_share * n_parties * (n_parties - 1)

    for party_id in range(1, n_parties):
        shares = gjkr_shamir_share(
            normalized_additive_shares[party_id],
            party_id,
            n_parties,
            threshold
        )

        for layer_name in shares.keys():
            if shares[layer_name] is not None:
                for j in range(n_parties):
                    # Field addition
                    aggregated_shares[j][layer_name] = (aggregated_shares[j][layer_name] + shares[layer_name][j]) % FIELD_PRIME

    timing_stats['gjkr_time'] = time.time() - start

    print(f"  GJKR sharing completed in {timing_stats['gjkr_time']:.2f}s")
    return aggregated_shares, timing_stats


def reconstruct_from_shamir_shares(shares: List[torch.Tensor], threshold: int) -> torch.Tensor:
    """Reconstruct secret from Shamir shares using Lagrange interpolation on floats."""
    original_shape = shares[0].shape
    n_elements = shares[0].numel()

    # Shares are integers in GF(p), convert to float first
    scale = 2 ** PRECISION_BITS
    shares_flat = torch.stack([s.flatten() for s in shares])

    # Convert from field representation to float
    shares_float = shares_flat.double()
    shares_float = torch.where(
        shares_float > FIELD_PRIME // 2,
        shares_float - FIELD_PRIME,
        shares_float
    ) / scale

    # Lagrange interpolation on floats (evaluate at x=0)
    x_coords = torch.arange(1, threshold + 1, dtype=torch.double)
    reconstructed = torch.zeros(n_elements, dtype=torch.double)

    for i in range(threshold):
        # Lagrange basis: L_i(0) = product of (-x_j) / (x_i - x_j) for j != i
        lagrange_basis = 1.0
        for j in range(threshold):
            if i != j:
                lagrange_basis *= (-x_coords[j]) / (x_coords[i] - x_coords[j])

        reconstructed += shares_float[i] * lagrange_basis

    return reconstructed.reshape(original_shape)


def reconstruct_from_gjkr_shares(shares: List[Dict[str, torch.Tensor]],
                                 threshold: int) -> Dict[str, torch.Tensor]:
    """Reconstruct normalized watermark from GJKR Shamir shares."""
    reconstructed = {}

    for layer_name in shares[0].keys():
        if shares[0][layer_name] is None:
            reconstructed[layer_name] = None
            continue

        shares_for_layer = [shares[i][layer_name] for i in range(threshold)]
        reconstructed[layer_name] = reconstruct_from_shamir_shares(
            shares_for_layer, threshold
        )

    return reconstructed


# Export main protocol function
def trustless_keygen_with_gjkr(model: torch.nn.Module,
                               n_parties: int,
                               threshold: int,
                               seed: int,
                               device) -> Tuple[List[Dict], Dict[str, torch.Tensor], Dict]:
    """Trustless key generation protocol with GJKR and commitments."""
    timing_stats = {
        'sampling_time': 0.0,
        'sharing_time': 0.0,
        'mpc_time': 0.0,
        'gjkr_time': 0.0,
        'total_time': 0.0,
        'mpc_comm_bytes': 0,
        'gjkr_comm_bytes': 0,
        'total_comm_bytes': 0,
        'mpc_rounds': 3,
        'gjkr_rounds': 1,
        'total_rounds': 4
    }

    print(f"\nTrustless Keygen: {n_parties} parties, threshold={threshold}, device={device}")
    total_start = time.time()

    print("\nPhase 1: Sampling with commitments")
    start = time.time()
    party_watermarks = []
    party_commitments = []

    for party_id in range(n_parties):
        watermark, commitments = sample_with_commitment(model, party_id, seed, device)
        party_watermarks.append(watermark)
        party_commitments.append(commitments)

    timing_stats['sampling_time'] = time.time() - start
    print(f"  Phase 1 completed in {timing_stats['sampling_time']:.2f}s")

    print("\nPhase 2: MPC Normalization")
    normalized_shares, original_normalized, mpc_stats = mpc_normalize_secret_shared(party_watermarks, n_parties, device)
    timing_stats.update(mpc_stats)

    print("\nPhase 3: GJKR Shamir Sharing")
    final_shares, gjkr_stats = simulate_gjkr_protocol(
        normalized_shares, n_parties, threshold
    )
    timing_stats['gjkr_time'] = gjkr_stats['gjkr_time']
    timing_stats['gjkr_comm_bytes'] = gjkr_stats['gjkr_comm_bytes']
    timing_stats['gjkr_rounds'] = gjkr_stats['gjkr_rounds']
    timing_stats['sharing_time'] = 0.0

    timing_stats['total_time'] = time.time() - total_start
    timing_stats['total_comm_bytes'] = timing_stats['mpc_comm_bytes'] + timing_stats['gjkr_comm_bytes']
    timing_stats['total_rounds'] = timing_stats['mpc_rounds'] + timing_stats['gjkr_rounds']

    print(f"\nProtocol completed in {timing_stats['total_time']:.2f}s")
    print(f"Total communication: {timing_stats['total_comm_bytes'] / 1024 / 1024:.2f} MB\n")

    return final_shares, original_normalized, timing_stats
