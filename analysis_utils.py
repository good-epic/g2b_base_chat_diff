"""
Utility functions for analyzing CrossCoder features and decoder directions.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def find_interesting_directions(global_df, cosine_threshold=0.8, norm_threshold=0.1):
    """
    Find interesting decoder directions that have:
    1. Low cosine similarity (below threshold)
    2. Non-small norms in both base and IT models
    
    Args:
        global_df: DataFrame with global decoder stats
        cosine_threshold: Upper threshold for cosine similarity
        norm_threshold: Lower threshold for L2 norms
    
    Returns:
        DataFrame with filtered directions
    """
    # Filter by cosine similarity
    low_cos_df = global_df[global_df['cosine_similarity'] < cosine_threshold]
    
    # Further filter by non-small norms
    interesting_df = low_cos_df[
        (low_cos_df['l2_norm_base'] > norm_threshold) & 
        (low_cos_df['l2_norm_it'] > norm_threshold)
    ]
    
    # Sort by cosine similarity (ascending)
    interesting_df = interesting_df.sort_values('cosine_similarity')
    
    return interesting_df

def analyze_feature_occurrence(df, interesting_features=None):
    """
    Analyze how often specific features occur in the jokes dataset
    
    Args:
        df: DataFrame with crosscoder metrics
        interesting_features: Optional list of feature indices to specifically analyze
    
    Returns:
        DataFrame with feature occurrence statistics
    """
    # Extract rows with nonzero_features that are not None
    jokes_df = df[df['nonzero_features'].notna()]
    
    # Count feature occurrences
    feature_counts = {}
    
    for features in jokes_df['nonzero_features']:
        if isinstance(features, list):  # Make sure it's a list
            for feat_idx in features:
                if feat_idx in feature_counts:
                    feature_counts[feat_idx] += 1
                else:
                    feature_counts[feat_idx] = 1
    
    # Convert to DataFrame
    feature_df = pd.DataFrame({
        'feature_index': list(feature_counts.keys()),
        'occurrence_count': list(feature_counts.values()),
        'occurrence_percentage': [count / len(jokes_df) * 100 for count in feature_counts.values()]
    })
    
    # Sort by occurrence count (descending)
    feature_df = feature_df.sort_values('occurrence_count', ascending=False)
    
    # If specific interesting features provided, add a flag
    if interesting_features is not None:
        feature_df['is_interesting'] = feature_df['feature_index'].isin(interesting_features)
    
    return feature_df

def plot_decoder_stats(global_df, save_dir='plots'):
    """
    Create various plots to visualize decoder statistics
    
    Args:
        global_df: DataFrame with global decoder stats
        save_dir: Directory to save plots
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Histogram of cosine similarities
    plt.figure(figsize=(10, 6))
    plt.hist(global_df['cosine_similarity'], bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Cosine Similarities Between Base and IT Decoder Directions')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.axvline(np.mean(global_df['cosine_similarity']), color='red', linestyle='dashed', 
                linewidth=1, label=f'Mean: {np.mean(global_df["cosine_similarity"]):.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cosine_similarity_histogram.png'), dpi=300)
    plt.close()
    
    # 2. Scatter plot of base vs IT L2 norms with cosine similarity as color
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(global_df['l2_norm_base'], global_df['l2_norm_it'], 
               c=global_df['cosine_similarity'], cmap='coolwarm', alpha=0.7)
    plt.colorbar(sc, label='Cosine Similarity')
    plt.title('L2 Norms of Base vs IT Decoder Directions')
    plt.xlabel('Base Model L2 Norm')
    plt.ylabel('IT Model L2 Norm')
    plt.grid(True, alpha=0.3)
    # Add diagonal line
    max_val = max(global_df['l2_norm_base'].max(), global_df['l2_norm_it'].max())
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'l2_norm_scatter.png'), dpi=300)
    plt.close()
    
    # 3. Histogram of L2 norm ratios (IT/Base) with log scale
    plt.figure(figsize=(10, 6))
    # Filter out infinite ratios
    valid_ratios = global_df[global_df['norm_ratio_l2'] < 1000]['norm_ratio_l2']
    plt.hist(valid_ratios, bins=50, color='lightgreen', edgecolor='black')
    plt.title('Distribution of L2 Norm Ratios (IT/Base)')
    plt.xlabel('L2 Norm Ratio (IT/Base)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.axvline(np.median(valid_ratios), color='red', linestyle='dashed', 
                linewidth=1, label=f'Median: {np.median(valid_ratios):.4f}')
    plt.axvline(1.0, color='black', linestyle='dashed', 
                linewidth=1, label='Equal Norms')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'l2_norm_ratio_histogram.png'), dpi=300)
    plt.close()
    
    # 4. Scatter plot of cosine similarity vs norm ratio with delta_norm as color
    plt.figure(figsize=(10, 8))
    # Filter extreme values for better visualization
    plot_df = global_df[(global_df['norm_ratio_l2'] < 5) & (global_df['norm_ratio_l2'] > 0.2)]
    sc = plt.scatter(plot_df['cosine_similarity'], plot_df['norm_ratio_l2'], 
               c=plot_df['delta_norm'], cmap='viridis', alpha=0.7)
    plt.colorbar(sc, label='Delta Norm')
    plt.title('Relationship Between Direction Change and Magnitude Ratio')
    plt.xlabel('Cosine Similarity (directional change)')
    plt.ylabel('L2 Norm Ratio IT/Base (magnitude ratio)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1, color='r', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cosine_vs_ratio_scatter.png'), dpi=300)
    plt.close()
    
    # 5. Heatmap of interesting features (low cosine, high norms)
    interesting_df = find_interesting_directions(global_df)
    
    if len(interesting_df) > 0:
        # Create a correlation-style matrix for visualization
        feat_indices = interesting_df['feature_index'].values
        num_features = len(feat_indices)
        
        if num_features > 50:
            # If too many features, take a subset
            interesting_df = interesting_df.head(50)
            feat_indices = interesting_df['feature_index'].values
            num_features = len(feat_indices)
        
        # Prepare data for heatmap
        heatmap_data = np.zeros((num_features, 4))
        
        for i, idx in enumerate(feat_indices):
            row = interesting_df[interesting_df['feature_index'] == idx]
            heatmap_data[i, 0] = row['cosine_similarity'].values[0]
            heatmap_data[i, 1] = row['l2_norm_base'].values[0]
            heatmap_data[i, 2] = row['l2_norm_it'].values[0]
            heatmap_data[i, 3] = row['delta_norm'].values[0]
        
        # Create custom colormaps
        cmap_cos = LinearSegmentedColormap.from_list('cos_cmap', ['crimson', 'white'])
        cmap_norm = LinearSegmentedColormap.from_list('norm_cmap', ['white', 'darkblue'])
        cmap_delta = LinearSegmentedColormap.from_list('delta_cmap', ['lightblue', 'white', 'lightgreen'])
        
        fig, axes = plt.subplots(1, 4, figsize=(20, num_features/2), sharey=True)
        
        # Feature indices as y-tick labels
        yticks = [f"F{idx}" for idx in feat_indices]
        
        # Cosine similarity heatmap
        sns.heatmap(heatmap_data[:, 0:1], ax=axes[0], cmap=cmap_cos, 
                   cbar_kws={'label': 'Value'}, yticklabels=yticks)
        axes[0].set_title('Cosine Similarity')
        axes[0].set_xticklabels(['cos_sim'])
        
        # L2 Norm Base heatmap
        sns.heatmap(heatmap_data[:, 1:2], ax=axes[1], cmap=cmap_norm, 
                   cbar_kws={'label': 'Value'})
        axes[1].set_title('L2 Norm Base')
        axes[1].set_xticklabels(['l2_base'])
        axes[1].set_yticklabels([])
        
        # L2 Norm IT heatmap
        sns.heatmap(heatmap_data[:, 2:3], ax=axes[2], cmap=cmap_norm, 
                   cbar_kws={'label': 'Value'})
        axes[2].set_title('L2 Norm IT')
        axes[2].set_xticklabels(['l2_it'])
        axes[2].set_yticklabels([])
        
        # Delta Norm heatmap
        sns.heatmap(heatmap_data[:, 3:4], ax=axes[3], cmap=cmap_delta, 
                   cbar_kws={'label': 'Value'}, center=0.5)
        axes[3].set_title('Delta Norm')
        axes[3].set_xticklabels(['delta_norm'])
        axes[3].set_yticklabels([])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'interesting_features_heatmap.png'), dpi=300)
        plt.close()

def plot_kl_divergence_analysis(df, save_dir='plots'):
    """
    Analyze and plot KL divergence distributions and correlations
    
    Args:
        df: DataFrame with crosscoder metrics
        save_dir: Directory to save plots
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract rows with KL divergence values
    kl_df = df[df['kl_divergence'].notna()]
    
    # 1. Histogram of KL divergences
    plt.figure(figsize=(10, 6))
    plt.hist(kl_df['kl_divergence'], bins=30, color='salmon', edgecolor='black')
    plt.title('Distribution of KL Divergences Between Base and IT Models')
    plt.xlabel('KL Divergence')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.axvline(np.mean(kl_df['kl_divergence']), color='blue', linestyle='dashed', 
                linewidth=1, label=f'Mean: {np.mean(kl_df["kl_divergence"]):.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kl_divergence_histogram.png'), dpi=300)
    plt.close()
    
    # Add a new column for number of features
    kl_df['num_features'] = kl_df['nonzero_features'].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    
    # 2. Scatter plot of number of nonzero features vs KL divergence
    plt.figure(figsize=(10, 6))
    plt.scatter(kl_df['num_features'], kl_df['kl_divergence'], alpha=0.6, color='purple')
    plt.title('Relationship Between Number of Active Features and KL Divergence')
    plt.xlabel('Number of Nonzero Features')
    plt.ylabel('KL Divergence')
    plt.grid(True, alpha=0.3)
    
    # Add regression line
    z = np.polyfit(kl_df['num_features'], kl_df['kl_divergence'], 1)
    p = np.poly1d(z)
    plt.plot(kl_df['num_features'], p(kl_df['num_features']), "r--", alpha=0.8)
    
    # Add correlation coefficient
    corr = np.corrcoef(kl_df['num_features'], kl_df['kl_divergence'])[0,1]
    plt.text(0.05, 0.95, f'Correlation: {corr:.4f}', transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'features_vs_kl_scatter.png'), dpi=300)
    plt.close()

def generate_feature_report(global_df, feature_occurrence_df, top_n=50, save_dir='reports'):
    """
    Generate a report of the most interesting features based on multiple criteria
    
    Args:
        global_df: DataFrame with global decoder stats
        feature_occurrence_df: DataFrame with feature occurrence stats
        top_n: Number of features to include
        save_dir: Directory to save the report
    
    Returns:
        DataFrame with the most interesting active features
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Merge the dataframes
    merged_df = global_df.merge(feature_occurrence_df, on='feature_index', how='left')
    
    # Fill NaN values for features that never occurred
    merged_df['occurrence_count'] = merged_df['occurrence_count'].fillna(0)
    merged_df['occurrence_percentage'] = merged_df['occurrence_percentage'].fillna(0)
    
    # Generate interesting feature lists based on different criteria
    reports = {
        'lowest_cosine_sim': merged_df.sort_values('cosine_similarity').head(top_n),
        'highest_norm_change': merged_df.sort_values(by='norm_ratio_l2', ascending=False).head(top_n),
        'most_occurring': merged_df.sort_values(by='occurrence_count', ascending=False).head(top_n),
        'interesting_active': merged_df[
            (merged_df['cosine_similarity'] < 0.8) & 
            (merged_df['l2_norm_base'] > 0.1) & 
            (merged_df['l2_norm_it'] > 0.1) & 
            (merged_df['occurrence_count'] > 0)
        ].sort_values('cosine_similarity').head(top_n)
    }
    
    # Save each report
    for name, report_df in reports.items():
        report_df.to_csv(os.path.join(save_dir, f'{name}_features.csv'), index=False)
    
    # Return the combined report of all interesting active features
    return reports.get('interesting_active', pd.DataFrame())  # Return empty DataFrame if key doesn't exist