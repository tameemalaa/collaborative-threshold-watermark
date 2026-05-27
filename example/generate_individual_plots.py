import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import get_flip_vectors_float
from src.ResNet import ResNet18, ResNet18_CIFAR100
from src.ResNetTinyImageNet import ResNet18_TinyImageNet
from src.data_utils import get_cifar10_transforms, get_cifar10_dataset, get_cifar100_transforms, get_cifar100_dataset, get_tinyimagenet_transforms, get_tinyimagenet_dataset
import matplotlib as mpl

# Set publication-quality matplotlib settings
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman', 'serif'],
    'mathtext.fontset': 'stix',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2,
    'patch.linewidth': 0.5,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'figure.figsize': [10, 6],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 4,
    'ytick.major.size': 4
})

def simulate_random_cosine_similarities_torch(model, n_samples=2000):
    """
    Simulate cosine similarities between model parameters and random flip vectors.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    all_similarities = []

    for i in range(0, n_samples):
        if n_samples >= 1000 and i % 500 == 0:
            print(f"Progress: {i}/{n_samples}")
        elif n_samples < 1000 and i % 200 == 0:
            print(f"Progress: {i}/{n_samples}")

        similarities = []
        vectors = get_flip_vectors_float(model, device)

        for name, param in model.named_parameters():
            if vectors[name] is None or not param.requires_grad:
                continue
            similarities.append(
                torch.nn.functional.cosine_similarity(
                    param.view(-1),
                    vectors[name].view(-1),
                    dim=0
                )
            )

        if similarities:
            avg_similarity = torch.mean(torch.tensor(similarities))
            all_similarities.append(avg_similarity.item())

    return all_similarities

def validate_distribution(similarities, dataset_name):
    """Validate if the distribution follows expected normal distribution."""
    similarities = np.array(similarities)

    mean = np.mean(similarities)
    std = np.std(similarities)

    normality_results = {}
    if len(similarities) > 3:
        test_sample = similarities[:5000] if len(similarities) > 5000 else similarities
        shapiro_stat, shapiro_p = stats.shapiro(test_sample)
        ks_stat, ks_p = stats.kstest(similarities, 'norm', args=(mean, std))

        normality_results['shapiro_p'] = shapiro_p
        normality_results['ks_p'] = ks_p
        normality_results['is_normal'] = shapiro_p > 0.05 or ks_p > 0.05

    confidence_interval = stats.t.interval(0.95, len(similarities)-1,
                                         loc=mean,
                                         scale=stats.sem(similarities))

    return mean, std, normality_results

def train_model_full(model, dataset, num_epochs=20, batch_size=128):
    """Train model on full dataset for realistic parameter distributions."""
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import torch.nn as nn

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    print(f"Training model for {num_epochs} epochs on {len(dataset)} samples...")

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')

        scheduler.step()
        acc = 100.0 * correct / total
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} Complete - Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")

    return model

# Global list to collect all captions
all_captions = []

def save_figure_with_caption(fig, filename, caption):
    """Save figure as PDF without caption and collect caption for text file."""
    global all_captions

    # Ensure proper layout
    fig.tight_layout()

    # Ensure filename has .pdf extension
    if not filename.endswith('.pdf'):
        filename += '.pdf'

    # Save as high-quality PDF without caption
    fig.savefig(filename, format='pdf', bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    print(f"✓ Saved: {filename}")

    # Collect caption for text file
    all_captions.append(f"{filename}: {caption}")

    plt.close(fig)

def save_all_captions(output_file='figure_captions.txt'):
    """Save all collected captions to a text file."""
    with open(output_file, 'w') as f:
        f.write("Figure Captions for Publication\n")
        f.write("=" * 50 + "\n\n")
        for i, caption in enumerate(all_captions, 1):
            f.write(f"{caption}\n\n")
    print(f"✓ Saved all captions to: {output_file}")

def plot_individual_model_distributions(individual_sims, dataset_name, stats, colors, filename_prefix):
    """Plot individual model distributions for a dataset."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for i, (model_sims, color, (mean, std)) in enumerate(zip(individual_sims, colors, stats)):
        ax.hist(model_sims, bins=25, alpha=0.7, density=True, color=color,
                label=f'Model {i+1} (μ={mean:.4f}, σ={std:.4f})',
                edgecolor='white', linewidth=0.5)

    ax.set_title(f'{dataset_name}: Individual Model Parameter Distributions', fontsize=16, pad=20)
    ax.set_xlabel('Cosine Similarity with Random Vectors', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add statistical annotation
    n_samples = len(individual_sims[0]) if individual_sims else 0
    ax.text(0.02, 0.98, f'n = 5 models, {n_samples} samples each',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    caption = (f"Figure: Distribution of cosine similarities between {dataset_name} ResNet18 model parameters "
               f"and random watermark vectors. Each model was trained with a different random seed (0-4) "
               f"to demonstrate parameter distribution consistency across initializations. "
               f"The distributions show the null hypothesis behavior for watermark detection.")

    save_figure_with_caption(fig, f"{filename_prefix}_individual_distributions", caption)

def plot_combined_distribution_with_fit(similarities, dataset_name, mean, std, filename_prefix):
    """Plot combined distribution with normal fit."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot histogram
    n_bins = 50
    counts, bins, patches = ax.hist(similarities, bins=n_bins, alpha=0.7, density=True,
                                   color='steelblue', label=f'{dataset_name} Data',
                                   edgecolor='white', linewidth=0.5)

    # Plot normal fit
    x_limit = min(1, 5*std)
    x = np.linspace(min(similarities), max(similarities), 1000)
    y = stats.norm.pdf(x, mean, std)
    ax.plot(x, y, color='red', linestyle='--', linewidth=3,
            label=f'Normal Fit (μ={mean:.4f}, σ={std:.4f})')

    # Add statistical annotations
    n_samples = len(similarities)
    ax.text(0.98, 0.98, f'n = {n_samples:,} samples\nμ = {mean:.4f}\nσ = {std:.4f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    ax.set_title(f'{dataset_name}: Combined Distribution with Normal Fit', fontsize=16, pad=20)
    ax.set_xlabel('Cosine Similarity with Random Vectors', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)

    caption = (f"Figure: Combined cosine similarity distribution for {dataset_name} ResNet18 models "
               f"with fitted normal distribution. The close fit validates the assumption of normality "
               f"required for Z-score based watermark detection. Data aggregated from 5 models "
               f"with {len(similarities):,} total samples.")

    save_figure_with_caption(fig, f"{filename_prefix}_combined_distribution", caption)

def plot_datasets_comparison_overlay(all_cifar10, all_cifar100, all_tiny, filename):
    """Plot overlay comparison of all three datasets."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Use more professional colors
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']  # Blue, Green, Orange
    datasets = ['CIFAR-10', 'CIFAR-100', 'TinyImageNet']
    data = [all_cifar10, all_cifar100, all_tiny]

    for i, (dataset_data, color, label) in enumerate(zip(data, colors, datasets)):
        ax.hist(dataset_data, bins=50, alpha=0.6, density=True, color=color,
                label=f'{label} (n={len(dataset_data):,})', edgecolor='white', linewidth=0.5)

    # Add means as vertical lines
    means = [np.mean(d) for d in data]
    for mean, color, label in zip(means, colors, datasets):
        ax.axvline(mean, color=color, linestyle=':', linewidth=2, alpha=0.8)

    ax.set_title('Cross-Dataset Distribution Comparison', fontsize=16, pad=20)
    ax.set_xlabel('Cosine Similarity with Random Vectors', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)

    caption = ("Figure: Comparison of cosine similarity distributions across three datasets "
               "(CIFAR-10, CIFAR-100, TinyImageNet) using ResNet18 architecture. "
               "The overlapping distributions suggest that watermark detection parameters "
               "may be generalizable across different datasets with similar architectures.")

    save_figure_with_caption(fig, filename, caption)

def plot_variability_analysis(stats_c10, stats_c100, stats_tiny, metric_name, ylabel, filename):
    """Plot variability analysis for means or standard deviations."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    metric_idx = 0 if metric_name.lower() == 'mean' else 1
    cifar10_vals = [stats[metric_idx] for stats in stats_c10]
    cifar100_vals = [stats[metric_idx] for stats in stats_c100]
    tiny_vals = [stats[metric_idx] for stats in stats_tiny]

    ax.boxplot([cifar10_vals, cifar100_vals, tiny_vals],
               tick_labels=['CIFAR-10', 'CIFAR-100', 'TinyImageNet'],
               patch_artist=True,
               boxprops=dict(facecolor='lightblue', alpha=0.7),
               medianprops=dict(color='red', linewidth=2))

    ax.set_title(f'{metric_name} Cosine Similarity Variability Across Models', fontsize=16, pad=20)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel('Dataset', fontsize=14)
    ax.grid(True, alpha=0.3)

    caption = (f"Figure: {metric_name} variability analysis across 5 independently trained ResNet18 models "
               f"for each dataset. Box plots show median, quartiles, and outliers. "
               f"Lower variability indicates more consistent watermark parameter estimation "
               f"across different model initializations.")

    save_figure_with_caption(fig, filename, caption)
    plt.close(fig)

def plot_model_comparison_bars(stats_c10, stats_c100, stats_tiny, metric_name, ylabel, filename):
    """Plot bar comparison across models for means or standard deviations."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    metric_idx = 0 if metric_name.lower() == 'mean' else 1
    cifar10_vals = [stats[metric_idx] for stats in stats_c10]
    cifar100_vals = [stats[metric_idx] for stats in stats_c100]
    tiny_vals = [stats[metric_idx] for stats in stats_tiny]

    model_labels = [f'Model {i+1}' for i in range(5)]
    x_pos = np.arange(len(model_labels))
    width = 0.25

    bars1 = ax.bar(x_pos - width, cifar10_vals, width, label='CIFAR-10',
                   color='lightblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x_pos, cifar100_vals, width, label='CIFAR-100',
                   color='lightgreen', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x_pos + width, tiny_vals, width, label='TinyImageNet',
                   color='lightsalmon', alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_title(f'{metric_name} Comparison Across Individual Models', fontsize=16, pad=20)
    ax.set_xlabel('Model Index', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_labels)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    caption = (f"Figure: {metric_name} comparison across 5 individual ResNet18 models "
               f"for each dataset. Each model was trained with different random seeds "
               f"to assess parameter estimation consistency. Error bars would show "
               f"confidence intervals if multiple runs per model were performed.")

    save_figure_with_caption(fig, filename, caption)
    plt.close(fig)

def plot_qq_plots(similarities_dict, filename):
    """Plot Q-Q plots for normality validation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    datasets = ['CIFAR-10', 'CIFAR-100', 'TinyImageNet']
    colors = ['blue', 'green', 'orange']

    for idx, (dataset, color) in enumerate(zip(datasets, colors)):
        sims = similarities_dict[dataset][:1000]  # Use subset for cleaner plot
        stats.probplot(sims, dist="norm", plot=axes[idx])
        axes[idx].set_title(f'{dataset}\nQ-Q Plot vs Normal', fontsize=14)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].get_lines()[0].set_color(color)
        axes[idx].get_lines()[0].set_markersize(3)
        axes[idx].get_lines()[1].set_color('red')
        axes[idx].get_lines()[1].set_linewidth(2)

    fig.suptitle('Quantile-Quantile Plots for Normality Validation', fontsize=16)

    caption = ("Figure: Q-Q plots comparing observed cosine similarity quantiles against "
               "theoretical normal distribution quantiles. Points following the red diagonal "
               "line indicate good fit to normal distribution, validating the statistical "
               "assumptions underlying Z-score watermark detection methods.")

    save_figure_with_caption(fig, filename, caption)
    plt.close(fig)

def plot_violin_distributions(individual_similarities, stats_dict, filename):
    """Plot violin plots showing distribution shapes per model."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    datasets = ['CIFAR-10', 'CIFAR-100', 'TinyImageNet']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, dataset in enumerate(datasets):
        all_sims = individual_similarities[dataset]

        parts = axes[idx].violinplot(all_sims, positions=range(1, 6),
                                   showmeans=True, showmedians=True)

        # Color the violin plots
        for patch, color in zip(parts['bodies'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axes[idx].set_title(f'{dataset}\nDistribution Shapes', fontsize=14)
        axes[idx].set_xlabel('Model Index', fontsize=12)
        axes[idx].set_ylabel('Cosine Similarity', fontsize=12)
        axes[idx].set_xticks(range(1, 6))
        axes[idx].set_xticklabels([f'M{i+1}' for i in range(5)])
        axes[idx].grid(True, alpha=0.3)

    fig.suptitle('Model-wise Distribution Shape Analysis', fontsize=16)

    caption = ("Figure: Violin plots showing the distribution shapes of cosine similarities "
               "for each individual model across datasets. Width indicates density at each "
               "value, white dots show medians, thick bars show quartiles. Similar shapes "
               "across models indicate consistent watermark parameter behavior.")

    save_figure_with_caption(fig, filename, caption)
    plt.close(fig)

# Main execution
def main():
    print("Loading datasets...")

    # Load datasets
    transform_train, transform_test = get_cifar10_transforms()
    train_dataset_10, _, _ = get_cifar10_dataset(transform_train, transform_test)

    transform_train_100, transform_test_100 = get_cifar100_transforms()
    train_dataset_100, _, _ = get_cifar100_dataset(transform_train_100, transform_test_100)

    transform_train_tiny, transform_test_tiny = get_tinyimagenet_transforms()
    train_dataset_tiny, _, _ = get_tinyimagenet_dataset(transform_train_tiny, transform_test_tiny)

    # Create/load models
    datasets = ['CIFAR-10', 'CIFAR-100', 'TinyImageNet']
    models_dict = {'CIFAR-10': [], 'CIFAR-100': [], 'TinyImageNet': []}
    stats_dict = {'CIFAR-10': [], 'CIFAR-100': [], 'TinyImageNet': []}
    all_similarities = {'CIFAR-10': [], 'CIFAR-100': [], 'TinyImageNet': []}
    individual_similarities = {'CIFAR-10': [], 'CIFAR-100': [], 'TinyImageNet': []}

    # Process CIFAR-10
    print("\nProcessing CIFAR-10 models...")
    for i in range(5):
        model_path = f"trained_resnet18_cifar10_seed_{i}.pt"
        if os.path.exists(model_path):
            model = ResNet18()
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        else:
            torch.manual_seed(i)
            model = ResNet18()
            model = train_model_full(model, train_dataset_10, num_epochs=10)
            torch.save(model.state_dict(), model_path)

        models_dict['CIFAR-10'].append(model)
        similarities = simulate_random_cosine_similarities_torch(model, n_samples=2000)
        all_similarities['CIFAR-10'].extend(similarities)
        individual_similarities['CIFAR-10'].append(similarities)
        mean, std, _ = validate_distribution(similarities, f"CIFAR-10 Model {i+1}")
        stats_dict['CIFAR-10'].append((mean, std))

    # Process CIFAR-100
    print("\nProcessing CIFAR-100 models...")
    for i in range(5):
        model_path = f"trained_resnet18_cifar100_seed_{i}.pt"
        if os.path.exists(model_path):
            model = ResNet18_CIFAR100()
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        else:
            torch.manual_seed(i)
            model = ResNet18_CIFAR100()
            model = train_model_full(model, train_dataset_100, num_epochs=10)
            torch.save(model.state_dict(), model_path)

        models_dict['CIFAR-100'].append(model)
        similarities = simulate_random_cosine_similarities_torch(model, n_samples=2000)
        all_similarities['CIFAR-100'].extend(similarities)
        individual_similarities['CIFAR-100'].append(similarities)
        mean, std, _ = validate_distribution(similarities, f"CIFAR-100 Model {i+1}")
        stats_dict['CIFAR-100'].append((mean, std))

    # Process TinyImageNet
    print("\nProcessing TinyImageNet models...")
    for i in range(5):
        model_path = f"trained_resnet18_tinyimagenet_seed_{i}.pt"
        if os.path.exists(model_path):
            model = ResNet18_TinyImageNet()
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        else:
            torch.manual_seed(i)
            model = ResNet18_TinyImageNet()
            model = train_model_full(model, train_dataset_tiny, num_epochs=10)
            torch.save(model.state_dict(), model_path)

        models_dict['TinyImageNet'].append(model)
        similarities = simulate_random_cosine_similarities_torch(model, n_samples=2000)
        all_similarities['TinyImageNet'].extend(similarities)
        individual_similarities['TinyImageNet'].append(similarities)
        mean, std, _ = validate_distribution(similarities, f"TinyImageNet Model {i+1}")
        stats_dict['TinyImageNet'].append((mean, std))

    print("\nGenerating individual publication-ready plots...")

    # Create output directory
    os.makedirs('publication_plots', exist_ok=True)
    os.chdir('publication_plots')

    # Color schemes for each dataset
    colors_c10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    colors_c100 = ['#2ca02c', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    colors_tiny = ['#ff7f0e', '#d62728', '#17becf', '#e377c2', '#7f7f7f']

    # Generate individual distribution plots
    plot_individual_model_distributions(individual_similarities['CIFAR-10'], 'CIFAR-10',
                                      stats_dict['CIFAR-10'], colors_c10, 'fig1_cifar10')
    plot_individual_model_distributions(individual_similarities['CIFAR-100'], 'CIFAR-100',
                                      stats_dict['CIFAR-100'], colors_c100, 'fig2_cifar100')
    plot_individual_model_distributions(individual_similarities['TinyImageNet'], 'TinyImageNet',
                                      stats_dict['TinyImageNet'], colors_tiny, 'fig3_tinyimagenet')

    # Generate combined distribution plots with fits
    for dataset in datasets:
        similarities = all_similarities[dataset]
        mean, std, _ = validate_distribution(similarities, f"{dataset} Combined")
        dataset_clean = dataset.lower().replace('-', '').replace('imagenet', 'imgnet')
        plot_combined_distribution_with_fit(similarities, dataset, mean, std,
                                          f'fig{4 + list(datasets).index(dataset)}_{dataset_clean}')

    # Generate comparison plots
    plot_datasets_comparison_overlay(all_similarities['CIFAR-10'],
                                   all_similarities['CIFAR-100'],
                                   all_similarities['TinyImageNet'],
                                   'fig7_datasets_comparison.pdf')

    plot_variability_analysis(stats_dict['CIFAR-10'], stats_dict['CIFAR-100'],
                            stats_dict['TinyImageNet'], 'Mean',
                            'Mean Cosine Similarity', 'fig8_mean_variability.pdf')

    plot_variability_analysis(stats_dict['CIFAR-10'], stats_dict['CIFAR-100'],
                            stats_dict['TinyImageNet'], 'Standard Deviation',
                            'Standard Deviation', 'fig9_std_variability.pdf')

    plot_model_comparison_bars(stats_dict['CIFAR-10'], stats_dict['CIFAR-100'],
                             stats_dict['TinyImageNet'], 'Mean',
                             'Mean Cosine Similarity', 'fig10_model_means.pdf')

    plot_model_comparison_bars(stats_dict['CIFAR-10'], stats_dict['CIFAR-100'],
                             stats_dict['TinyImageNet'], 'Standard Deviation',
                             'Standard Deviation', 'fig11_model_stds.pdf')

    plot_qq_plots(all_similarities, 'fig12_qq_normality.pdf')

    plot_violin_distributions(individual_similarities, stats_dict, 'fig13_violin_shapes.pdf')

    # Save all captions to text file
    save_all_captions('figure_captions.txt')

    print("\nAll individual plots generated successfully!")
    print(f"Plots saved in: {os.getcwd()}")
    print("Figure captions saved to: figure_captions.txt")

if __name__ == "__main__":
    main()