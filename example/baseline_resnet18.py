import os
import torch
from tqdm import tqdm
import wandb
import copy
import csv
from src.ClientResNet18 import ClientResNet18
from src.ResNet import ResNet18
from src.utils import get_flip_vectors, set_seed, get_device, get_cosine_similarity_model, fedavg, save_plot
from src.data_utils import get_cifar10_transforms, get_cifar10_dataset, get_client_subsets, get_client_loaders, get_cifar10_dataloaders


seed = 0
global_batch_size = 2048
K = 4  # number of clients
batch_size_per_client = global_batch_size // K
initial_lr = 0.001  # initial learning rate
c = 0.1            # small constant to scale watermark
num_steps = 300      # total training steps
log_interval = 1
plot_interval = 1
beta = 0.9             # EMA decay for gradient norm
grad_norm_ema = 0.0    # Initialize EMA estimate
mean = 0.0
std = 0.01


device = get_device()
set_seed(seed)
print("Using device:", device)

wandb.init(
    project="baseline_resnet18",
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

model = ResNet18()
model = model.to(device)

# ----------- CLIENT FLIP VECTORS (UNIQUE FOR EACH CLIENT) -------------
client_flip_vectors = [get_flip_vectors(model, device) for _ in range(K)]
clients = [ClientResNet18(model, device, initial_lr, client_flip_vectors[i], c, beta) for i in range(K)]
transform_train, transform_test = get_cifar10_transforms()
train_dataset, validation_dataset, test_dataset = get_cifar10_dataset(transform_train, transform_test)
client_subsets = get_client_subsets(train_dataset, num_clients=K)
client_loaders = get_client_loaders(client_subsets, batch_size=batch_size_per_client)
train_loader, validation_loader, test_loader = get_cifar10_dataloaders(train_dataset, validation_dataset, test_dataset, batch_size=batch_size_per_client)

# -------------------------
# Results Directory
# -------------------------
results_dir = f"results_seed{seed}"
os.makedirs(results_dir, exist_ok=True)

# -------------------------
# Initialize Tracking Variables
# -------------------------
param_size = clients[0].get_number_of_parameters()
initial_acc = clients[0].evaluate(test_loader)
initial_val_acc = clients[0].evaluate(validation_loader)

print(f"Our model has {param_size} parameters.")
print(f"Initial Test Accuracy: {initial_acc:.2f}%")
print(f"Initial Validation Accuracy: {initial_val_acc:.2f}%")

# -------------------------
# Training Loop with Z-Score Tracking
# -------------------------
steps_logged = [0]
accuracy_history = [initial_acc]
validation_accuracy_history = [initial_val_acc]
losses_history = []
mean_zscore_history = []
highest_validation_accuracy = 0.0
highest_validation_accuracy_step = 0
inttial_cos_sim = [] 
initial_zscores = []
for client_idx, client in enumerate(clients):
    mean_cos_sim = get_cosine_similarity_model(model, client_flip_vectors[client_idx])
    inttial_cos_sim.append(mean_cos_sim)
    z_score = (mean_cos_sim - mean) / std
    initial_zscores.append(z_score)
client_zscore_history = [[initial_zscores[i]] for i in range(K)]
client_cosine_history = [[inttial_cos_sim[i]] for i in range(K)]  # If you want to keep cosine for debugging

# Calculate initial mean z-score
initial_mean_zscore = sum(initial_zscores) / K
mean_zscore_history.append(initial_mean_zscore)

wandb.log({
    "step":   0,
    "global_accuracy": initial_acc,
    "global_validation_accuracy": initial_val_acc,
})

global_parameters = model.state_dict()

for step in tqdm(range(num_steps), desc="Training"):
    model.train()
    avg_loss = 0
    avg_val_t = 0
    avg_current_mean_grad_norm = 0
    local_parameters = []
    bias_norm = 0

    for client_idx, client in enumerate(clients):
        loss, val_t, grad_norm_ema, grad_bias_norm = client.train_with_adding_bias(client_loaders[client_idx])
        avg_loss += loss
        avg_val_t += val_t
        avg_current_mean_grad_norm += grad_norm_ema
        bias_norm += grad_bias_norm
        params = client.get_parameters()
        local_parameters.append(params)
    global_parameters = fedavg(local_parameters)
    model.load_state_dict(global_parameters)
    for client_idx, client in enumerate(clients):
        client.set_parameters(global_parameters)
        mean_cos_sim = get_cosine_similarity_model(model, client_flip_vectors[client_idx])
        client_cosine_history[client_idx].append(mean_cos_sim)
        z_score = (mean_cos_sim - mean) / std
        client_zscore_history[client_idx].append(z_score)
    accuracy = clients[0].evaluate(test_loader)
    validation_accuracy = clients[0].evaluate(validation_loader)
    avg_loss = avg_loss / K 
    print(f"Step: {step+1},"
            f" Val(t): {avg_val_t:.5g}, meanGradNorm: {avg_current_mean_grad_norm:.4f}, "
            f"Test Accuracy: {accuracy:.2f}%, Validation Accuracy: {validation_accuracy:.2f}%")
    steps_logged.append(step+1)
    accuracy_history.append(accuracy)
    validation_accuracy_history.append(validation_accuracy)
    losses_history.append(avg_loss)
    # Calculate mean z-score across all clients
    current_zscores = [client_zscore_history[client_idx][-1] for client_idx in range(K)]
    mean_zscore = sum(current_zscores) / K
    mean_zscore_history.append(mean_zscore)
    
    wandb.log({
        "step": step + 1,
        "global_avg_loss": avg_loss,
        "global_accuracy": accuracy,
        "global_validation_accuracy": validation_accuracy,
        "global_avg_val_t": avg_val_t,
        "global_avg_grad_norm": avg_current_mean_grad_norm,
        "bias_norm": bias_norm,
        "mean_zscore": mean_zscore,
    })

    # Save the highest validation accuracy model
    if validation_accuracy > highest_validation_accuracy:
        highest_validation_accuracy = validation_accuracy
        highest_validation_accuracy_step = step + 1
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

# Export results to CSV files
results = {
    "steps": steps_logged,
    "accuracy": accuracy_history,
    "validation_accuracy": validation_accuracy_history,
    "losses": losses_history,
    "mean_zscore": mean_zscore_history,
}

for key, values in results.items():
    csv_filename = f"{results_dir}/{key}_K{K}_lr{initial_lr}_c{c}_steps{num_steps}_bs{batch_size_per_client}_seed{seed}.csv"
    with open(csv_filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([key])  # Write the header
        writer.writerows([[value] for value in values])  # Write the values
    print(f"{key.capitalize()} results saved to {csv_filename}")

for client_idx in range(K):
    # Plot z-scores
    save_plot(
        steps_logged,
        client_zscore_history[client_idx],
        "Training Step",
        "Z-Score",
        f"Client {client_idx + 1} Z-Score",
        f"{results_dir}/client_{client_idx + 1}_zscore_K{K}_lr{initial_lr}_c{c}_steps{num_steps}_bs{batch_size_per_client}_seed{seed}.pdf",
        "Z-Score",
        "purple",
        "-",
        "s"
    )

    # Plot cosine similarities
    save_plot(
        steps_logged,
        [cs.item() if isinstance(cs, torch.Tensor) else cs for cs in client_cosine_history[client_idx]],
        "Training Step",
        "Cosine Similarity",
        f"Client {client_idx + 1} Cosine Similarity",
        f"{results_dir}/client_{client_idx + 1}_cosine_similarity_K{K}_lr{initial_lr}_c{c}_steps{num_steps}_bs{batch_size_per_client}_seed{seed}.pdf",
        "Cosine Similarity",
        "blue",
        "-",
        "o"
    )

# Export z-scores and cosine similarities to CSV files
for client_idx in range(K):
    # Save z-scores
    zscore_csv_filename = f"{results_dir}/client_{client_idx + 1}_zscore_K{K}_lr{initial_lr}_c{c}_steps{num_steps}_bs{batch_size_per_client}_seed{seed}.csv"
    with open(zscore_csv_filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Step", "Z-Score"])  # Write the header
        writer.writerows(zip(steps_logged, client_zscore_history[client_idx]))  # Write the values
    print(f"Client {client_idx + 1} Z-Score results saved to {zscore_csv_filename}")

    # Save cosine similarities
    cosine_csv_filename = f"{results_dir}/client_{client_idx + 1}_cosine_similarity_K{K}_lr{initial_lr}_c{c}_steps{num_steps}_bs{batch_size_per_client}_seed{seed}.csv"
    with open(cosine_csv_filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Step", "Cosine Similarity"])  # Write the header
        writer.writerows(zip(steps_logged, [cs.item() if isinstance(cs, torch.Tensor) else cs for cs in client_cosine_history[client_idx]]))  # Write the values
    print(f"Client {client_idx + 1} Cosine Similarity results saved to {cosine_csv_filename}")

# Save the final model
final_model_filename = f"{results_dir}/cifar10_fedavg_baseline_K{K}_lr{initial_lr}_c{c}_steps{num_steps}_bs{batch_size_per_client}_seed{seed}.pt"
torch.save(model.state_dict(), final_model_filename)
print(f"Final model saved to {final_model_filename}")

# Print the validation accuracy of the best model
print(f"Highest validation accuracy model achieved at step {highest_validation_accuracy_step} with validation accuracy {highest_validation_accuracy:.2f}%")

wandb.finish()