import random
import torch
import matplotlib.pyplot as plt


def save_plot(x, y, xlabel, ylabel, title, filename, legend_label, color, linestyle, marker):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=legend_label, color=color, linestyle=linestyle, marker=marker)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{filename}", format='pdf', dpi=600)
    plt.close()

def fedavg(client_parameters):
    averaged_params = {}
    for name in client_parameters[0].keys():
        if 'num_batches_tracked' in name:
            averaged_params[name] = torch.max(torch.stack([client_params[name] 
                                  for client_params in client_parameters]))
        else:
            averaged_params[name] = torch.stack([client_params[name] 
                                  for client_params in client_parameters]).mean(dim=0)
    return averaged_params


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_cosine_similarity_model(target, source):
    cos_sims = []
    for name, param in target.named_parameters():
        if param.requires_grad:
            cos_sim = torch.nn.functional.cosine_similarity(
                param.view(-1),
                source[name].view(-1),
                dim=0
            )
            cos_sims.append(cos_sim)
    mean_cos_sim = 0.0
    if len(cos_sims) > 0:
        mean_cos_sim = sum(cos_sims).item() / len(cos_sims)
    return mean_cos_sim

def get_cosine_similarity_gradient(target, source):
    cos_sims = []
    for name, param in target.items():
        if name in source and source[name] is not None:
            param = param.to(source[name].device)  # Ensure param is on the same device as source[name]
            if param.device != source[name].device:
                raise RuntimeError(f"Device mismatch: param is on {param.device}, source[{name}] is on {source[name].device}")
            cos_sim = torch.nn.functional.cosine_similarity(
                param.view(-1),
                source[name].view(-1),
                dim=0
            )
            cos_sims.append(cos_sim)
    mean_cos_sim = 0.0
    if len(cos_sims) > 0:
        mean_cos_sim = sum(cos_sims).item() / len(cos_sims)
    return mean_cos_sim

def get_flip_vectors(model, device):
    flip_vectors = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            x = torch.randint_like(param, low=0, high=2) * 2 - 1
            flip_vectors[name] = (x / x.norm()).to(device)
        else:
            flip_vectors[name] = None
    return flip_vectors

def get_flip_vectors_same_as_parameters(model, device):
    flip_vectors = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_values = param.clone().detach()
            norm = param_values.norm()
            flip_vectors[name] = (param_values / norm).to(device)
        else:
            flip_vectors[name] = None
    return flip_vectors

def get_flip_vectors_same_as_gradients(gradients, device):
    flip_vectors = {}
    for name, param in gradients.items():
        if param is not None:
            param_values = param.clone().detach()
            norm = param_values.norm()
            flip_vectors[name] = (param_values / norm).to(device)
        else:
            flip_vectors[name] = None
    return flip_vectors

def get_flip_vectors_float(model, device):
    flip_vectors = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            x = torch.randn_like(param)
            flip_vectors[name] = (x / x.norm()).to(device)
        else:
            flip_vectors[name] = None
    return flip_vectors


def create_additive_shares(flip_vectors, num_shares=2):
    """
    Create additive secret shares of the flip vectors.
    
    Args:
        flip_vectors: Dictionary of normalized flip vectors
        num_shares: Number of shares to create
        
    Returns:
        List of dictionaries, each containing a share for each parameter
    """
    shares = [{} for _ in range(num_shares)]
    for name, vec in flip_vectors.items():
        if vec is not None:
            vec_cpu  = vec.to('cpu')
            # Create random shares that sum to the original vector
            param_shares = [torch.randn_like(vec_cpu) for _ in range(num_shares-1)]
            # Last share ensures they all sum to the original vector
            final_share = vec_cpu.clone()
            for share in param_shares:
                final_share -= share
            
            param_shares.append(final_share)
            
            # Store each share in the corresponding dictionary
            for i, share in enumerate(param_shares):
                shares[i][name] = share
        else:
            for i in range(num_shares):
                shares[i][name] = None
    return shares

def reconstruct_from_shares(shares):
    """
    Reconstruct the original flip vectors from shares.
    
    Args:
        shares: List of dictionaries containing the shares
        
    Returns:
        Dictionary of reconstructed flip vectors
    """
    reconstructed = {}
    
    # Get the parameter names from the first share
    if not shares or not shares[0]:
        return reconstructed
    
    for name in shares[0].keys():
        if all(share[name] is not None for share in shares):
            # Sum all shares to get the original vector
            reconstructed[name] = sum(share[name] for share in shares)
        else:
            reconstructed[name] = None
    
    return reconstructed


def safe_save_model(model, filepath):
    """Safely save model state dict with error handling"""
    try:
        torch.save(model.state_dict(), filepath)
        return True
    except Exception as e:
        print(f"Error saving model to {filepath}: {e}")
        return False

def safe_save_checkpoint(checkpoint_data, filepath):
    """Safely save checkpoint with error handling"""
    try:
        torch.save(checkpoint_data, filepath)
        return True
    except Exception as e:
        print(f"Error saving checkpoint to {filepath}: {e}")
        return False

def test_additive_secret_sharing(flip_vectors, reconstructed):
    """
    Test the additive secret sharing scheme for flip vectors with minimal testing.
    
    Args:
        model: The neural network model
        device: Device to place the tensors on
    """
    
    # Verify reconstruction
    success = True
    for name, original in flip_vectors.items():
        if original is not None:
            diff = torch.max(torch.abs(original.to('cpu') - reconstructed[name])).item()
            if diff > 1e-3:
                print(f"Error: Parameter {name} differs by {diff}")
                success = False
    
    if success:
        print("Success: All parameters reconstructed correctly")
    else:
        print("Failure: Some parameters weren't reconstructed correctly")