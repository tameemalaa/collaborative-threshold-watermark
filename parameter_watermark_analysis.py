"""
Parameter Watermark Analysis for Federated Learning Models

This script provides comprehensive analysis of watermark distribution across 
ResNet18 parameters using both global and layer-specific statistical baselines.

Required Inputs:
- watermarked_model_path: Path to trained model state dict (.pt file)
- flip_vectors_path: Path to watermark flip vectors (.pt file) 
- baseline_checkpoint_path: Path to clean model for baseline estimation (.pt file)
- global_stats: Pre-computed global statistics (mean, std) for the dataset
- dataset_type: 'cifar10' or 'cifar100'
- output_dir: Directory for saving analysis results

Usage Example:
# Set paths at the top of the script, then run
analyzer = ParameterWatermarkAnalyzer(...)
analyzer.run_complete_analysis()
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from pathlib import Path
import json

# Global statistics from your research (smol.txt)
GLOBAL_STATS = {
    'cifar10': {'mean': 0, 'std': 0.01},
    'cifar100': {'mean': 0, 'std': 0.0088}
}

class ParameterWatermarkAnalyzer:
    def __init__(self, watermarked_model_path, flip_vectors_path, baseline_checkpoint_path, 
                 global_stats, dataset_type, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.watermarked_model = torch.load(watermarked_model_path, map_location='cpu')
        self.flip_vectors = torch.load(flip_vectors_path, map_location='cpu')
        self.baseline_checkpoint = torch.load(baseline_checkpoint_path, map_location='cpu')
        
        # Statistics
        self.global_stats = global_stats[dataset_type]
        self.dataset_type = dataset_type
        
        # Analysis results storage
        self.parameter_results = {}
        self.layer_baselines = {}
        self.layer_results = {}
        
        print(f"Loaded watermarked model: {len(self.watermarked_model)} parameters")
        print(f"Loaded flip vectors: {len(self.flip_vectors)} vectors")
        print(f"Using {dataset_type} global stats: mean={self.global_stats['mean']:.6f}, std={self.global_stats['std']:.6f}")

    def cosine_similarity(self, tensor1, tensor2):
        """Compute cosine similarity between two tensors"""
        return torch.nn.functional.cosine_similarity(
            tensor1.view(-1), tensor2.view(-1), dim=0
        ).item()

    def generate_random_flip_vectors_like(self, flip_vectors):
        """Generate random flip vectors with same structure as original"""
        random_flip = {}
        for name, vec in flip_vectors.items():
            if vec is not None:
                random_vec = torch.randn_like(vec)
                random_flip[name] = random_vec / random_vec.norm()
            else:
                random_flip[name] = None
        return random_flip

    def get_layer_name(self, param_name):
        """Extract layer name from parameter name"""
        if 'conv1' in param_name:
            return 'conv1'
        elif 'layer1' in param_name:
            return 'layer1'
        elif 'layer2' in param_name:
            return 'layer2'
        elif 'layer3' in param_name:
            return 'layer3'
        elif 'layer4' in param_name:
            return 'layer4'
        elif 'linear' in param_name:
            return 'linear'
        else:
            return 'other'

    def get_parameter_type(self, param_name):
        """Classify parameter type"""
        if 'conv' in param_name and 'weight' in param_name:
            return 'conv_weight'
        elif 'bn' in param_name and 'weight' in param_name:
            return 'bn_weight'
        elif 'bn' in param_name and 'bias' in param_name:
            return 'bn_bias'
        elif 'linear' in param_name and 'weight' in param_name:
            return 'linear_weight'
        elif 'linear' in param_name and 'bias' in param_name:
            return 'linear_bias'
        else:
            return 'other'

    def estimate_layer_baselines(self, num_samples=100):
        """Estimate layer-specific statistical baselines using random flip vectors"""
        print("Estimating layer-specific baselines...")
        
        layer_cosines = defaultdict(list)
        
        for sample_idx in range(num_samples):
            if sample_idx % 20 == 0:
                print(f"Processing sample {sample_idx}/{num_samples}")
            
            # Generate random flip vectors
            random_flip = self.generate_random_flip_vectors_like(self.flip_vectors)
            
            # Compute cosine similarities for each parameter
            for name, param in self.baseline_checkpoint.items():
                if random_flip[name] is None:
                    continue
                
                cos_sim = self.cosine_similarity(param, random_flip[name])
                layer_name = self.get_layer_name(name)
                layer_cosines[layer_name].append(cos_sim)
        
        # Compute statistics for each layer
        self.layer_baselines = {}
        for layer_name, cosines in layer_cosines.items():
            self.layer_baselines[layer_name] = {
                'mean': np.mean(cosines),
                'std': np.std(cosines),
                'n_samples': len(cosines),
                'min': np.min(cosines),
                'max': np.max(cosines)
            }
        
        print("Layer-specific baselines estimated:")
        for layer, stats in self.layer_baselines.items():
            print(f"  {layer}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, n={stats['n_samples']}")
        
        return self.layer_baselines

    def compute_parameter_analysis(self):
        """Compute dual statistical analysis for all parameters"""
        print("Computing parameter-level analysis...")
        
        self.parameter_results = {}
        
        for name, param in self.watermarked_model.items():
            if self.flip_vectors[name] is None:
                continue
            
            # Compute cosine similarity
            cos_sim = self.cosine_similarity(param, self.flip_vectors[name])
            
            # Global z-score
            global_zscore = (cos_sim - self.global_stats['mean']) / self.global_stats['std']
            
            # Layer-specific z-score
            layer_name = self.get_layer_name(name)
            if layer_name in self.layer_baselines:
                layer_baseline = self.layer_baselines[layer_name]
                layer_zscore = (cos_sim - layer_baseline['mean']) / layer_baseline['std']
            else:
                layer_zscore = None
            
            # Store results
            self.parameter_results[name] = {
                'cosine_sim': cos_sim,
                'global_zscore': global_zscore,
                'layer_zscore': layer_zscore,
                'layer': layer_name,
                'param_type': self.get_parameter_type(name),
                'param_shape': list(param.shape),
                'param_size': param.numel()
            }
        
        print(f"Analysis completed for {len(self.parameter_results)} parameters")
        return self.parameter_results

    def analyze_by_parameter_type(self):
        """Analyze watermark strength by parameter type"""
        print("Analyzing by parameter type...")
        
        type_results = defaultdict(lambda: {
            'cosines': [], 'global_zscores': [], 'layer_zscores': [], 'param_names': []
        })
        
        for name, result in self.parameter_results.items():
            param_type = result['param_type']
            type_results[param_type]['cosines'].append(result['cosine_sim'])
            type_results[param_type]['global_zscores'].append(result['global_zscore'])
            if result['layer_zscore'] is not None:
                type_results[param_type]['layer_zscores'].append(result['layer_zscore'])
            type_results[param_type]['param_names'].append(name)
        
        # Compute summary statistics
        type_summary = {}
        for param_type, data in type_results.items():
            type_summary[param_type] = {
                'count': len(data['cosines']),
                'mean_cosine': np.mean(data['cosines']),
                'std_cosine': np.std(data['cosines']),
                'mean_global_zscore': np.mean(data['global_zscores']),
                'mean_layer_zscore': np.mean(data['layer_zscores']) if data['layer_zscores'] else None,
                'significant_global': sum(1 for z in data['global_zscores'] if abs(z) > 2.0),
                'significant_layer': sum(1 for z in data['layer_zscores'] if abs(z) > 2.0) if data['layer_zscores'] else 0
            }
        
        return type_summary, type_results

    def analyze_by_layer(self):
        """Analyze watermark strength by ResNet layer"""
        print("Analyzing by layer...")
        
        layer_results = defaultdict(lambda: {
            'cosines': [], 'global_zscores': [], 'layer_zscores': [], 'param_names': []
        })
        
        for name, result in self.parameter_results.items():
            layer_name = result['layer']
            layer_results[layer_name]['cosines'].append(result['cosine_sim'])
            layer_results[layer_name]['global_zscores'].append(result['global_zscore'])
            if result['layer_zscore'] is not None:
                layer_results[layer_name]['layer_zscores'].append(result['layer_zscore'])
            layer_results[layer_name]['param_names'].append(name)
        
        # Compute summary statistics
        layer_summary = {}
        for layer_name, data in layer_results.items():
            layer_summary[layer_name] = {
                'count': len(data['cosines']),
                'mean_cosine': np.mean(data['cosines']),
                'std_cosine': np.std(data['cosines']),
                'mean_global_zscore': np.mean(data['global_zscores']),
                'mean_layer_zscore': np.mean(data['layer_zscores']) if data['layer_zscores'] else None,
                'max_global_zscore': np.max(data['global_zscores']),
                'max_layer_zscore': np.max(data['layer_zscores']) if data['layer_zscores'] else None,
                'significant_global': sum(1 for z in data['global_zscores'] if abs(z) > 2.0),
                'significant_layer': sum(1 for z in data['layer_zscores'] if abs(z) > 2.0) if data['layer_zscores'] else 0
            }
        
        self.layer_results = layer_summary
        return layer_summary, layer_results

    def find_critical_parameters(self, top_k=20):
        """Identify most critical parameters for watermark detection"""
        print("Finding critical parameters...")
        
        # Sort by global z-score
        global_ranking = sorted(self.parameter_results.items(), 
                               key=lambda x: abs(x[1]['global_zscore']), reverse=True)[:top_k]
        
        # Sort by layer z-score
        layer_ranking = sorted([item for item in self.parameter_results.items() if item[1]['layer_zscore'] is not None],
                              key=lambda x: abs(x[1]['layer_zscore']), reverse=True)[:top_k]
        
        return global_ranking, layer_ranking

    def analyze_tensor_geometry(self):
        """Analyze relationship between tensor size and watermark strength"""
        print("Analyzing tensor geometry...")
        
        sizes = []
        cosines = []
        global_zscores = []
        layer_zscores = []
        dimensions = []
        
        for name, result in self.parameter_results.items():
            sizes.append(result['param_size'])
            cosines.append(result['cosine_sim'])
            global_zscores.append(result['global_zscore'])
            if result['layer_zscore'] is not None:
                layer_zscores.append(result['layer_zscore'])
            dimensions.append(len(result['param_shape']))
        
        # Compute correlations
        size_cosine_corr = np.corrcoef(sizes, cosines)[0, 1]
        size_global_z_corr = np.corrcoef(sizes, global_zscores)[0, 1]
        if layer_zscores:
            size_layer_z_corr = np.corrcoef(sizes[:len(layer_zscores)], layer_zscores)[0, 1]
        else:
            size_layer_z_corr = None
        
        return {
            'size_cosine_correlation': size_cosine_corr,
            'size_global_z_correlation': size_global_z_corr,
            'size_layer_z_correlation': size_layer_z_corr,
            'sizes': sizes,
            'cosines': cosines,
            'global_zscores': global_zscores,
            'layer_zscores': layer_zscores,
            'dimensions': dimensions
        }

    def create_visualizations(self):
        """Generate all visualization plots"""
        print("Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Parameter type analysis
        self.plot_parameter_type_analysis()
        
        # 2. Layer analysis  
        self.plot_layer_analysis()
        
        # 3. Critical parameters
        self.plot_critical_parameters()
        
        # 4. Tensor geometry analysis
        self.plot_tensor_geometry()
        
        # 5. Statistical method comparison
        self.plot_statistical_comparison()
        
        print("All visualizations saved!")

    def plot_parameter_type_analysis(self):
        """Create parameter type analysis plots"""
        type_summary, type_results = self.analyze_by_parameter_type()
        
        # Box plot of cosine similarities by parameter type
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Cosine similarities
        data_for_boxplot = []
        labels_for_boxplot = []
        for param_type, data in type_results.items():
            data_for_boxplot.extend(data['cosines'])
            labels_for_boxplot.extend([param_type] * len(data['cosines']))
        
        df = pd.DataFrame({'Parameter Type': labels_for_boxplot, 'Cosine Similarity': data_for_boxplot})
        sns.boxplot(data=df, x='Parameter Type', y='Cosine Similarity', ax=axes[0, 0])
        axes[0, 0].set_title('Cosine Similarity by Parameter Type')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Global z-scores
        data_for_boxplot = []
        labels_for_boxplot = []
        for param_type, data in type_results.items():
            data_for_boxplot.extend(data['global_zscores'])
            labels_for_boxplot.extend([param_type] * len(data['global_zscores']))
        
        df = pd.DataFrame({'Parameter Type': labels_for_boxplot, 'Global Z-Score': data_for_boxplot})
        sns.boxplot(data=df, x='Parameter Type', y='Global Z-Score', ax=axes[0, 1])
        axes[0, 1].set_title('Global Z-Scores by Parameter Type')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].axhline(y=2.0, color='r', linestyle='--', alpha=0.7, label='95% Confidence')
        axes[0, 1].axhline(y=-2.0, color='r', linestyle='--', alpha=0.7)
        axes[0, 1].legend()
        
        # Layer z-scores
        data_for_boxplot = []
        labels_for_boxplot = []
        for param_type, data in type_results.items():
            if data['layer_zscores']:
                data_for_boxplot.extend(data['layer_zscores'])
                labels_for_boxplot.extend([param_type] * len(data['layer_zscores']))
        
        if data_for_boxplot:
            df = pd.DataFrame({'Parameter Type': labels_for_boxplot, 'Layer Z-Score': data_for_boxplot})
            sns.boxplot(data=df, x='Parameter Type', y='Layer Z-Score', ax=axes[1, 0])
            axes[1, 0].set_title('Layer-Specific Z-Scores by Parameter Type')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].axhline(y=2.0, color='r', linestyle='--', alpha=0.7, label='95% Confidence')
            axes[1, 0].axhline(y=-2.0, color='r', linestyle='--', alpha=0.7)
            axes[1, 0].legend()
        
        # Summary statistics bar plot
        param_types = list(type_summary.keys())
        mean_cosines = [type_summary[pt]['mean_cosine'] for pt in param_types]
        
        axes[1, 1].bar(param_types, mean_cosines)
        axes[1, 1].set_title('Mean Cosine Similarity by Parameter Type')
        axes[1, 1].set_ylabel('Mean Cosine Similarity')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_type_analysis.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'parameter_type_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_layer_analysis(self):
        """Create layer analysis plots"""
        layer_summary, layer_results = self.analyze_by_layer()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Layer-wise mean cosine similarity
        layers = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'linear']
        mean_cosines = [layer_summary.get(layer, {}).get('mean_cosine', 0) for layer in layers]
        
        axes[0, 0].bar(layers, mean_cosines)
        axes[0, 0].set_title('Mean Watermark Strength by Layer')
        axes[0, 0].set_ylabel('Mean Cosine Similarity')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Layer-wise global z-scores
        mean_global_zscores = [layer_summary.get(layer, {}).get('mean_global_zscore', 0) for layer in layers]
        max_global_zscores = [layer_summary.get(layer, {}).get('max_global_zscore', 0) for layer in layers]
        
        x = np.arange(len(layers))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, mean_global_zscores, width, label='Mean', alpha=0.8)
        axes[0, 1].bar(x + width/2, max_global_zscores, width, label='Max', alpha=0.8)
        axes[0, 1].set_title('Global Z-Scores by Layer')
        axes[0, 1].set_ylabel('Z-Score')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(layers, rotation=45)
        axes[0, 1].axhline(y=2.0, color='r', linestyle='--', alpha=0.7)
        axes[0, 1].legend()
        
        # Layer-wise layer-specific z-scores
        mean_layer_zscores = [layer_summary.get(layer, {}).get('mean_layer_zscore', 0) if layer_summary.get(layer, {}).get('mean_layer_zscore') is not None else 0 for layer in layers]
        max_layer_zscores = [layer_summary.get(layer, {}).get('max_layer_zscore', 0) if layer_summary.get(layer, {}).get('max_layer_zscore') is not None else 0 for layer in layers]
        
        axes[1, 0].bar(x - width/2, mean_layer_zscores, width, label='Mean', alpha=0.8)
        axes[1, 0].bar(x + width/2, max_layer_zscores, width, label='Max', alpha=0.8)
        axes[1, 0].set_title('Layer-Specific Z-Scores by Layer')
        axes[1, 0].set_ylabel('Layer Z-Score')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(layers, rotation=45)
        axes[1, 0].axhline(y=2.0, color='r', linestyle='--', alpha=0.7)
        axes[1, 0].legend()
        
        # Significant parameters count
        sig_global = [layer_summary.get(layer, {}).get('significant_global', 0) for layer in layers]
        sig_layer = [layer_summary.get(layer, {}).get('significant_layer', 0) for layer in layers]
        
        axes[1, 1].bar(x - width/2, sig_global, width, label='Global Method', alpha=0.8)
        axes[1, 1].bar(x + width/2, sig_layer, width, label='Layer Method', alpha=0.8)
        axes[1, 1].set_title('Significant Parameters by Layer')
        axes[1, 1].set_ylabel('Count (|z| > 2.0)')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(layers, rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'layer_analysis.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'layer_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_critical_parameters(self):
        """Plot critical parameters analysis"""
        global_ranking, layer_ranking = self.find_critical_parameters()
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        
        # Top parameters by global z-score
        param_names = [name.split('.')[-2] + '.' + name.split('.')[-1] for name, _ in global_ranking[:10]]
        global_zscores = [abs(result['global_zscore']) for _, result in global_ranking[:10]]
        
        axes[0].barh(param_names, global_zscores)
        axes[0].set_title('Top 10 Parameters by Global Z-Score')
        axes[0].set_xlabel('|Global Z-Score|')
        axes[0].axvline(x=2.0, color='r', linestyle='--', alpha=0.7, label='95% Confidence')
        axes[0].legend()
        
        # Top parameters by layer z-score
        if layer_ranking:
            param_names = [name.split('.')[-2] + '.' + name.split('.')[-1] for name, _ in layer_ranking[:10]]
            layer_zscores = [abs(result['layer_zscore']) for _, result in layer_ranking[:10]]
            
            axes[1].barh(param_names, layer_zscores)
            axes[1].set_title('Top 10 Parameters by Layer-Specific Z-Score')
            axes[1].set_xlabel('|Layer Z-Score|')
            axes[1].axvline(x=2.0, color='r', linestyle='--', alpha=0.7, label='95% Confidence')
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'critical_parameters.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'critical_parameters.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_tensor_geometry(self):
        """Plot tensor geometry analysis"""
        geometry_results = self.analyze_tensor_geometry()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Parameter size vs cosine similarity
        axes[0, 0].scatter(geometry_results['sizes'], geometry_results['cosines'], alpha=0.6)
        axes[0, 0].set_xlabel('Parameter Count')
        axes[0, 0].set_ylabel('Cosine Similarity')
        axes[0, 0].set_title(f'Parameter Size vs Watermark Strength\n(Correlation: {geometry_results["size_cosine_correlation"]:.3f})')
        axes[0, 0].set_xscale('log')
        
        # Parameter size vs global z-score
        axes[0, 1].scatter(geometry_results['sizes'], geometry_results['global_zscores'], alpha=0.6)
        axes[0, 1].set_xlabel('Parameter Count')
        axes[0, 1].set_ylabel('Global Z-Score')
        axes[0, 1].set_title(f'Parameter Size vs Global Z-Score\n(Correlation: {geometry_results["size_global_z_correlation"]:.3f})')
        axes[0, 1].set_xscale('log')
        axes[0, 1].axhline(y=2.0, color='r', linestyle='--', alpha=0.7)
        axes[0, 1].axhline(y=-2.0, color='r', linestyle='--', alpha=0.7)
        
        # Parameter size vs layer z-score
        if geometry_results['layer_zscores']:
            sizes_subset = geometry_results['sizes'][:len(geometry_results['layer_zscores'])]
            axes[1, 0].scatter(sizes_subset, geometry_results['layer_zscores'], alpha=0.6)
            axes[1, 0].set_xlabel('Parameter Count')
            axes[1, 0].set_ylabel('Layer Z-Score')
            axes[1, 0].set_title(f'Parameter Size vs Layer Z-Score\n(Correlation: {geometry_results["size_layer_z_correlation"]:.3f})')
            axes[1, 0].set_xscale('log')
            axes[1, 0].axhline(y=2.0, color='r', linestyle='--', alpha=0.7)
            axes[1, 0].axhline(y=-2.0, color='r', linestyle='--', alpha=0.7)
        
        # Tensor dimensions histogram
        axes[1, 1].hist(geometry_results['dimensions'], bins=range(1, max(geometry_results['dimensions'])+2), alpha=0.7)
        axes[1, 1].set_xlabel('Tensor Dimensions')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Distribution of Parameter Tensor Dimensions')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tensor_geometry_analysis.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'tensor_geometry_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_statistical_comparison(self):
        """Compare global vs layer-specific statistical methods"""
        # Get data for comparison
        global_zscores = []
        layer_zscores = []
        param_names = []
        
        for name, result in self.parameter_results.items():
            if result['layer_zscore'] is not None:
                global_zscores.append(result['global_zscore'])
                layer_zscores.append(result['layer_zscore'])
                param_names.append(name)
        
        if not global_zscores:
            print("No layer z-scores available for comparison")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Scatter plot comparison
        axes[0, 0].scatter(global_zscores, layer_zscores, alpha=0.6)
        axes[0, 0].set_xlabel('Global Z-Score')
        axes[0, 0].set_ylabel('Layer-Specific Z-Score')
        axes[0, 0].set_title('Global vs Layer-Specific Z-Scores')
        
        # Add reference lines
        max_val = max(max(global_zscores), max(layer_zscores))
        min_val = min(min(global_zscores), min(layer_zscores))
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='y=x')
        axes[0, 0].axhline(y=2.0, color='g', linestyle='--', alpha=0.5)
        axes[0, 0].axvline(x=2.0, color='g', linestyle='--', alpha=0.5)
        axes[0, 0].legend()
        
        # Compute correlation
        correlation = np.corrcoef(global_zscores, layer_zscores)[0, 1]
        axes[0, 0].text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=axes[0, 0].transAxes)
        
        # Distribution comparison
        axes[0, 1].hist(global_zscores, bins=30, alpha=0.5, label='Global Z-Scores', density=True)
        axes[0, 1].hist(layer_zscores, bins=30, alpha=0.5, label='Layer Z-Scores', density=True)
        axes[0, 1].set_xlabel('Z-Score')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Z-Score Distributions')
        axes[0, 1].axvline(x=2.0, color='r', linestyle='--', alpha=0.7)
        axes[0, 1].axvline(x=-2.0, color='r', linestyle='--', alpha=0.7)
        axes[0, 1].legend()
        
        # Significance agreement
        global_significant = [abs(z) > 2.0 for z in global_zscores]
        layer_significant = [abs(z) > 2.0 for z in layer_zscores]
        
        agreement = sum(g == l for g, l in zip(global_significant, layer_significant))
        agreement_rate = agreement / len(global_significant)
        
        # Create confusion matrix style plot
        both_sig = sum(g and l for g, l in zip(global_significant, layer_significant))
        global_only = sum(g and not l for g, l in zip(global_significant, layer_significant))
        layer_only = sum(not g and l for g, l in zip(global_significant, layer_significant))
        neither_sig = sum(not g and not l for g, l in zip(global_significant, layer_significant))
        
        categories = ['Both\nSignificant', 'Global\nOnly', 'Layer\nOnly', 'Neither\nSignificant']
        counts = [both_sig, global_only, layer_only, neither_sig]
        
        axes[1, 0].bar(categories, counts)
        axes[1, 0].set_title(f'Significance Agreement\n(Agreement Rate: {agreement_rate:.2%})')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Method sensitivity comparison
        global_sensitivity = sum(global_significant) / len(global_significant)
        layer_sensitivity = sum(layer_significant) / len(layer_significant)
        
        axes[1, 1].bar(['Global Method', 'Layer Method'], [global_sensitivity, layer_sensitivity])
        axes[1, 1].set_title('Detection Sensitivity Comparison')
        axes[1, 1].set_ylabel('Fraction of Significant Parameters')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'statistical_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'statistical_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self):
        """Save all analysis results to files"""
        print("Saving analysis results...")
        
        # Save parameter results
        param_df = pd.DataFrame.from_dict(self.parameter_results, orient='index')
        param_df.to_csv(self.output_dir / 'parameter_results.csv')
        
        # Save layer baselines
        with open(self.output_dir / 'layer_baselines.json', 'w') as f:
            json.dump(self.layer_baselines, f, indent=2)
        
        # Save layer results
        with open(self.output_dir / 'layer_results.json', 'w') as f:
            json.dump(self.layer_results, f, indent=2)
        
        # Save critical parameters
        global_ranking, layer_ranking = self.find_critical_parameters()
        
        global_critical = pd.DataFrame([
            {
                'parameter_name': name,
                'cosine_sim': result['cosine_sim'],
                'global_zscore': result['global_zscore'],
                'layer': result['layer'],
                'param_type': result['param_type']
            }
            for name, result in global_ranking
        ])
        global_critical.to_csv(self.output_dir / 'critical_parameters_global.csv', index=False)
        
        if layer_ranking:
            layer_critical = pd.DataFrame([
                {
                    'parameter_name': name,
                    'cosine_sim': result['cosine_sim'],
                    'layer_zscore': result['layer_zscore'],
                    'layer': result['layer'],
                    'param_type': result['param_type']
                }
                for name, result in layer_ranking
            ])
            layer_critical.to_csv(self.output_dir / 'critical_parameters_layer.csv', index=False)
        
        # Save summary report
        self.generate_summary_report()
        
        print(f"All results saved to {self.output_dir}")

    def generate_summary_report(self):
        """Generate a summary report of the analysis"""
        report = []
        report.append(f"# Parameter Watermark Analysis Report")
        report.append(f"Dataset: {self.dataset_type}")
        report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Global statistics
        report.append("## Global Statistics Used")
        report.append(f"Mean: {self.global_stats['mean']:.6f}")
        report.append(f"Std:  {self.global_stats['std']:.6f}")
        report.append("")
        
        # Layer baselines
        report.append("## Layer-Specific Baselines")
        for layer, stats in self.layer_baselines.items():
            report.append(f"{layer:>8}: mean={stats['mean']:>8.6f}, std={stats['std']:>8.6f}, n={stats['n_samples']:>4}")
        report.append("")
        
        # Parameter summary
        total_params = len(self.parameter_results)
        significant_global = sum(1 for r in self.parameter_results.values() if abs(r['global_zscore']) > 2.0)
        significant_layer = sum(1 for r in self.parameter_results.values() if r['layer_zscore'] is not None and abs(r['layer_zscore']) > 2.0)
        
        report.append("## Parameter Summary")
        report.append(f"Total Parameters Analyzed: {total_params}")
        report.append(f"Significant (Global Method):  {significant_global} ({significant_global/total_params:.1%})")
        report.append(f"Significant (Layer Method):   {significant_layer} ({significant_layer/total_params:.1%})")
        report.append("")
        
        # Layer analysis summary
        report.append("## Layer Analysis Summary")
        if hasattr(self, 'layer_results'):
            for layer in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'linear']:
                if layer in self.layer_results:
                    lr = self.layer_results[layer]
                    report.append(f"{layer:>8}: mean_cosine={lr['mean_cosine']:>7.5f}, "
                                f"global_z={lr['mean_global_zscore']:>6.2f}, "
                                f"layer_z={lr['mean_layer_zscore']:>6.2f if lr['mean_layer_zscore'] is not None else 'N/A':>6}")
        report.append("")
        
        # Critical parameters
        global_ranking, layer_ranking = self.find_critical_parameters(top_k=5)
        report.append("## Top 5 Critical Parameters (Global Z-Score)")
        for i, (name, result) in enumerate(global_ranking, 1):
            short_name = name.split('.')[-2] + '.' + name.split('.')[-1]
            report.append(f"{i}. {short_name:<20} z={result['global_zscore']:>6.2f} cos={result['cosine_sim']:>8.5f}")
        
        if layer_ranking:
            report.append("")
            report.append("## Top 5 Critical Parameters (Layer Z-Score)")
            for i, (name, result) in enumerate(layer_ranking, 1):
                short_name = name.split('.')[-2] + '.' + name.split('.')[-1]
                report.append(f"{i}. {short_name:<20} z={result['layer_zscore']:>6.2f} cos={result['cosine_sim']:>8.5f}")
        
        # Save report
        with open(self.output_dir / 'analysis_report.txt', 'w') as f:
            f.write('\n'.join(report))

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting complete parameter watermark analysis...")
        print("="*60)
        
        # Step 1: Estimate layer baselines
        self.estimate_layer_baselines()
        
        # Step 2: Compute parameter analysis
        self.compute_parameter_analysis()
        
        # Step 3: Run specialized analyses
        self.analyze_by_parameter_type()
        self.analyze_by_layer()
        self.find_critical_parameters()
        self.analyze_tensor_geometry()
        
        # Step 4: Create visualizations
        self.create_visualizations()
        
        # Step 5: Save results
        self.save_results()
        
        print("="*60)
        print("Analysis complete! Check the output directory for results.")
        print(f"Results saved to: {self.output_dir}")


# ====================================================================
# CONFIGURATION - Set your file paths here
# ====================================================================

# Set these paths to your actual files
WATERMARKED_MODEL_PATH = "results/final_model.pt"              # Your trained watermarked model
FLIP_VECTORS_PATH = "results/flip_vectors.pt"                  # Your watermark flip vectors  
BASELINE_CHECKPOINT_PATH = "results/clean_model.pt"            # Clean model for baseline estimation
DATASET_TYPE = "cifar10"                                       # 'cifar10' or 'cifar100'
OUTPUT_DIR = "analysis_results/cifar10"                                # Output directory for results

# ====================================================================

def main():
    """Main function to run the analysis"""
    print("Parameter Watermark Analysis")
    print("="*50)
    print(f"Watermarked Model: {WATERMARKED_MODEL_PATH}")
    print(f"Flip Vectors: {FLIP_VECTORS_PATH}")
    print(f"Baseline Checkpoint: {BASELINE_CHECKPOINT_PATH}")
    print(f"Dataset Type: {DATASET_TYPE}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("="*50)
    
    # Create analyzer
    analyzer = ParameterWatermarkAnalyzer(
        watermarked_model_path=WATERMARKED_MODEL_PATH,
        flip_vectors_path=FLIP_VECTORS_PATH,
        baseline_checkpoint_path=BASELINE_CHECKPOINT_PATH,
        global_stats=GLOBAL_STATS,
        dataset_type=DATASET_TYPE,
        output_dir=OUTPUT_DIR
    )
    
    # Run complete analysis
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()