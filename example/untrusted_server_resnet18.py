import os
import torch
from tqdm import tqdm
import wandb
import copy
import csv
from src.ClientResNet18 import ClientResNet18
from src.ResNet import ResNet18
from src.utils import get_flip_vectors, set_seed, get_device, get_cosine_similarity_model, get_cosine_similarity_gradient, fedavg, save_plot, get_flip_vectors_same_as_gradients, create_additive_shares, reconstruct_from_shares, test_additive_secret_sharing
from src.data_utils import get_cifar10_transforms, get_cifar10_dataset, get_client_subsets, get_client_loaders, get_cifar10_dataloaders
import pickle


# --- helper to track cosine between weight-Δ and watermark -------------
def delta_cos(model, base_state_dict, flip_vectors):
    """
    Cosine similarity between each parameter's *delta*
    (current weight – initial weight) and the corresponding
    watermark vector.  Returns the mean over all layers that
    have a flip-vector.
    """
    total, count = 0.0, 0
    for name, mark in flip_vectors.items():
        if mark is None:
            continue
        delta = (model.state_dict()[name] - base_state_dict[name]).view(-1).float()
        total += torch.nn.functional.cosine_similarity(delta, mark.view(-1), dim=0).item()
        count += 1
    return total / count if count else 0.0
# ----------------------------------------------------------------------

# -------------------------
# Hyperparameters   
# -------------------------
seed = 1
global_batch_size = 2048
K = 64 # number of clients
batch_size_per_client = global_batch_size // K
initial_lr = 0.001
c = 0.025 # small constant to scale watermark
num_steps = 200 # total training steps
log_interval = 1
plot_interval = 1
beta = 0.9 # EMA decay for gradient norm
grad_norm_ema = 0.0 # Initialize EMA estimate
mean = 0.0
std = 0.01

wandb.init(
    project="untrusted_resnet18_watermark",
    config={
        "K": K,
        "batch_size_per_client": batch_size_per_client,
        "initial_lr": initial_lr,
        "c": c,
        "num_steps": num_steps,
        "log_interval": log_interval,
        "plot_interval": plot_interval,
        "beta": beta,
        "grad_norm_ema": grad_norm_ema,
        "mean": mean,
        "std": std,
    },
)

print(f"Initial LR={initial_lr}")
print(f"Watermark scale constant c={c}")


device = get_device()
set_seed(seed)
print("Using device:", device)

model = ResNet18()
model = model.to(device)
initial_state = copy.deepcopy(model.state_dict())

# client_flip_vectors = [get_flip_vectors(model, device) for _ in range(K)]
flip_vectors = get_flip_vectors(model, device)
shares = create_additive_shares(flip_vectors, K)
test_additive_secret_sharing(flip_vectors,reconstruct_from_shares(shares))

clients = [ClientResNet18(model, device, initial_lr, shares[i], c, beta) for i in range(K)]
transform_train, transform_test = get_cifar10_transforms()
train_dataset, validation_dataset, test_dataset = get_cifar10_dataset(transform_train, transform_test)
client_subsets = get_client_subsets(train_dataset, num_clients=K)
client_loaders = get_client_loaders(client_subsets, batch_size=batch_size_per_client)
train_loader, validation_loader, test_loader = get_cifar10_dataloaders(train_dataset, validation_dataset, test_dataset, batch_size=batch_size_per_client)

client_index = 0  # Change this to export a different client's loader

# Create directory for exported data
os.makedirs(f"untrusted_cifar10_exported_data_cifar_10_seed{seed}", exist_ok=True)

# Export the client dataset subset and loader parameters
client_export_data = {
    "dataset_subset": client_subsets[client_index],
    "batch_size": batch_size_per_client,
    "shuffle": True,
}

# Export the test dataset and loader parameters
test_export_data = {
    "dataset": test_dataset,
    "batch_size": 128,
    "shuffle": False,
}

# # Save the client export data
# with open(f"untrusted_cifar10_exported_data_cifar_10/k{K}_C{c}_loader_data.pkl", "wb") as f:
#     pickle.dump(client_export_data, f)

# # Save the test export data
# with open(f"untrusted_cifar10_exported_data_cifar_10/k{K}_test_loader_data.pkl", "wb") as f:
#     pickle.dump(test_export_data, f)

print(f"Exported client {client_index} loader data")
print("Exported test loader data")
# -------------------------
# Results Directory
# -------------------------
results_dir = "unntrusted_cifar10_results"
os.makedirs(results_dir, exist_ok=True)

# -------------------------
# Initialize Tracking Variables
# -------------------------
param_size = clients[0].get_number_of_parameters()
initial_acc = clients[0].evaluate(test_loader)
initial_val_acc = clients[0].evaluate(validation_loader)
mean_cos_sim = get_cosine_similarity_model(model, flip_vectors)
t_mean_cos_sim = torch.tensor(mean_cos_sim)
intital_z_score = (mean_cos_sim - mean) / std

print(f"Our model has {param_size} parameters.")
print(f"Initial Test Accuracy: {initial_acc:.2f}%")
print(f"Initial Validation Accuracy: {initial_val_acc:.2f}%")
print(f"Initial Cosine Similarity: {mean_cos_sim:.6f}")
print(f"Initial Z-score: {intital_z_score:.6f}")

steps_logged = [0]
cosine_history = [mean_cos_sim]
accuracy_history = [initial_acc]
validation_accuracy_history = [initial_val_acc]
z_scores = [intital_z_score]
losses_history = []
attack_cosine_history = [0.0]
highest_validation_accuracy = 0.0
highest_validation_accuracy_step = 0
highest_validation_accuracy_z_score = None  # To store the z-score of the best model
global_parameters = model.state_dict()
# INITIALISE once, before the training loop
client_cosine_history = [[0.0] for _ in range(K)]   # 0.0 is the Δ-cos at step 0

for step in tqdm(range(num_steps), desc="Training"):
    model.train()
    avg_loss = 0
    scale_factor_all_clients = []
    local_parameters = []
    avg_val_t = 0
    total_current_mean_grad_norm=0
    for client_idx, client in enumerate(clients):
        loss ,val_t,current_mean_grad_norm,scale_factors = client.train_with_calcluting_bias(client_loaders[client_idx])
        avg_loss += loss
        avg_val_t += val_t
        total_current_mean_grad_norm +=current_mean_grad_norm
        scale_factor_all_clients.append(scale_factors)
    avg_scale_factor = {}
    pseudo_grad = {}
    for name in flip_vectors:
        if  flip_vectors[name] is not None: 
            if name not in avg_scale_factor:
                avg_scale_factor[name] = 0.0            
            for client_idx in range(K):
                avg_scale_factor[name] += scale_factor_all_clients[client_idx][name]
            avg_scale_factor[name] = avg_scale_factor[name]
    for client_idx in range(K):
        client_local_paramters = clients[client_idx].add_bias(avg_scale_factor)
        local_parameters.append(client_local_paramters)
    old_global_parameters = copy.deepcopy(global_parameters)
    global_parameters = fedavg(local_parameters)
    for client_idx, client in enumerate(clients):
        client.set_parameters(global_parameters)
    model.load_state_dict(global_parameters)
    
    for name in old_global_parameters.keys():
        pseudo_grad[name] = global_parameters[name].to(device).float() - old_global_parameters[name].to(device).float()
    if step == 0:   
        pseudo_grad_acc =  pseudo_grad
    else:
        for name in pseudo_grad_acc.keys():
            pseudo_grad_acc[name] += pseudo_grad[name]
    accuracy = clients[0].evaluate(test_loader)
    validation_accuracy = clients[0].evaluate(validation_loader)
    avg_loss = avg_loss / K
    print(f'Round {step+1}/{num_steps}, '
        f'Avg Loss: {avg_loss:.4f}, '
        f'Global Test Accuracy: {accuracy:.2f}%, '
        f'Global Validation Accuracy: {validation_accuracy:.2f}%')

    mean_cos_sim = get_cosine_similarity_model(model, flip_vectors)
    z_score = (mean_cos_sim - mean) / std
    attack_mean_cos_sim = get_cosine_similarity_gradient(pseudo_grad_acc, flip_vectors)

    print(f"Step: {step+1},"
          f"Val(t): {avg_val_t:.5g}, meanGradNorm: {total_current_mean_grad_norm:.4f}, "
          f"Test Accuracy: {accuracy:.2f}%, Validation Accuracy: {validation_accuracy:.2f}%, CosSim: {mean_cos_sim:.6f}, z-score: {z_score:.6f}")
    steps_logged.append(step+1)
    cosine_history.append(mean_cos_sim)
    accuracy_history.append(accuracy)
    validation_accuracy_history.append(validation_accuracy)
    z_scores.append(z_score)
    losses_history.append(avg_loss)
    attack_cosine_history.append(attack_mean_cos_sim)
    wandb.log({
        "step": step + 1,
        "cosine_similarity": mean_cos_sim,
        "global_avg_loss": avg_loss,
        "global_accuracy": accuracy,
        "global_validation_accuracy": validation_accuracy,
        "global_avg_val_t": val_t,
        "global_avg_grad_norm": total_current_mean_grad_norm,
        "z_score": z_score,
        "attack_cosine": attack_mean_cos_sim
    })

    # Save the highest validation accuracy model
    if validation_accuracy > highest_validation_accuracy:
        highest_validation_accuracy = validation_accuracy
        highest_validation_accuracy_step = step + 1
        highest_validation_accuracy_z_score = z_score  # Store the z-score of the best model
        model_filename = f"{results_dir}/highest_validation_accuracy_model_K{K}_lr{initial_lr}_c{c}_steps{num_steps}_bs{batch_size_per_client}_seed{seed}.pt"
        torch.save(model.state_dict(), model_filename)
        print(f"New highest validation accuracy model saved at step {highest_validation_accuracy_step} with validation accuracy {highest_validation_accuracy:.2f}%")

    # Save plots
    hyperparams = f"K{K}_lr{initial_lr}_c{c}_steps{num_steps}_bs{batch_size_per_client}_seed{seed}"
    save_plot(steps_logged[1:], losses_history, "Training Step", "Loss",
              "Training Loss", f"{results_dir}/training_loss_{hyperparams}.pdf", "Loss", "red", "-", "o")
    save_plot(steps_logged, accuracy_history, "Training Step", "Accuracy (%)",
              "Model Test Accuracy", f"{results_dir}/model_test_accuracy_{hyperparams}.pdf", "Test Accuracy", "green", "-", "x")
    save_plot(steps_logged, validation_accuracy_history, "Training Step", "Accuracy (%)",
              "Model Validation Accuracy", f"{results_dir}/model_validation_accuracy_{hyperparams}.pdf", "Validation Accuracy", "blue", "-", "x")
    save_plot(steps_logged, [cs.item() if isinstance(cs, torch.Tensor) else cs for cs in cosine_history],
              "Training Step", "Cosine Similarity", "Watermark Alignment",
              f"{results_dir}/watermark_alignment_{hyperparams}.pdf", "Cosine Similarity", "blue", "-", "o")
    save_plot(steps_logged, [z.item() if isinstance(z, torch.Tensor) else z for z in z_scores],
              "Training Step", "Z-Score", "Statistical Significance",
              f"{results_dir}/statistical_significance_{hyperparams}.pdf", "Z-Score", "purple", "-", "s")
    save_plot(steps_logged, attack_cosine_history, "Training Step", "Attack Cosine Similarity",
              "Accumulated Gradient Cosine Similarity with Actual Key", f"{results_dir}/attack_cosine_similarity_{hyperparams}.pdf",
              "Accumulated Gradient Similarity", "orange", "-", "s")

# Export results to CSV files
results = {
    "steps": steps_logged,
    "accuracy": accuracy_history,
    "validation_accuracy": validation_accuracy_history,
    "cosine_similarity": [cs.item() if isinstance(cs, torch.Tensor) else cs for cs in cosine_history],
    "z_scores": [z.item() if isinstance(z, torch.Tensor) else z for z in z_scores],
    "losses": losses_history,
    "attack_cosine_similarity": attack_cosine_history,
}

for key, values in results.items():
    csv_filename = f"{results_dir}/{key}_K{K}_lr{initial_lr}_c{c}_steps{num_steps}_bs{batch_size_per_client}_seed{seed}.csv"
    with open(csv_filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([key])  # Write the header
        writer.writerows([[value] for value in values])  # Write the values
    print(f"{key.capitalize()} results saved to {csv_filename}")

# Save the final model and flip vectors
final_model_filename = f"{results_dir}/cifar10_fedavg_watermark_K{K}_lr{initial_lr}_c{c}_steps{num_steps}_bs{batch_size_per_client}_seed{seed}.pt"
torch.save(model.state_dict(), final_model_filename)
print(f"Final model saved to {final_model_filename}")

flip_vectors_filename = f"{results_dir}/cifar10_flip_vectors_K{K}_lr{initial_lr}_c{c}_steps{num_steps}_bs{batch_size_per_client}_seed{seed}.pt"
torch.save(flip_vectors, flip_vectors_filename)
print(f"Flip vectors saved to {flip_vectors_filename}")

attack_key = get_flip_vectors_same_as_gradients(pseudo_grad_acc, device)
torch.save(attack_key, f"cifar10_attack_key_K{K}_lr{initial_lr}_c{c}_steps{num_steps}_bs{batch_size_per_client}_seed{seed}.pt")
# Print the z-score and validation accuracy of the best model
print(f"Highest validation accuracy model achieved at step {highest_validation_accuracy_step} with validation accuracy {highest_validation_accuracy:.2f}% and z-score {highest_validation_accuracy_z_score:.6f}")

wandb.finish()