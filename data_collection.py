"""
Memory-efficient data collection that creates CPU-based DataFrames
for later analysis, while efficiently handling the large crosscoder dimensions.
"""

import os
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import gc
import time

def process_single_joke(joke, joke_idx, gemma_2, gemma_2_it, crosscoder, token_index=-5):
    """
    Process a single joke with minimal memory usage.
    
    Args:
        joke: The joke string
        joke_idx: Index of the joke
        gemma_2: Base model
        gemma_2_it: Instruction-tuned model
        crosscoder: Crosscoder model
        token_index: Token position to analyze
    
    Returns:
        Tuple of (joke_data, feature_data)
    """
    # Setup cosine similarity function
    cos_sim = nn.CosineSimilarity(dim=0)
    
    # Initialize containers
    joke_data = {}
    feature_data = []
    
    try:
        # --- First get activations only ---
        with gemma_2.trace(joke):
            l13_act_base = gemma_2.model.layers[13].output[0][:, token_index].save()
            gemma_2.model.layers[13].output.stop()
        
        with gemma_2_it.trace(joke):
            l13_act_it = gemma_2_it.model.layers[13].output[0][:, token_index].save()
            gemma_2_it.model.layers[13].output.stop()
        
        # --- Now get logits for KL calculation ---
        with th.no_grad():
            with gemma_2.trace(joke):
                logits_base = gemma_2.model.output[0][:, token_index].save()
                gemma_2.model.output.stop()
            
            with gemma_2_it.trace(joke):
                logits_it = gemma_2_it.model.output[0][:, token_index].save()
                gemma_2_it.model.output.stop()
            
            # Calculate KL divergence immediately
            kl_div = calculate_kl_divergence(logits_base, logits_it).item()
            
            # Clean up logits to save memory
            del logits_base, logits_it
            th.cuda.empty_cache()
        
        # Prepare input for crosscoder
        crosscoder_input = th.cat([l13_act_base, l13_act_it], dim=0).unsqueeze(0)
        
        # Clean up original activations
        del l13_act_base, l13_act_it
        
        # Get features from crosscoder (but not reconstruction to save memory)
        with th.no_grad():
            _, features = crosscoder(crosscoder_input, output_features=True)
            
            # Find non-zero features immediately
            nonzero_mask = features.abs() > 1e-4
            nonzero_idx = nonzero_mask.nonzero().cpu().numpy()
            nonzero_feature_indices = [int(ni[1]) for ni in nonzero_idx]
            
            # Clean up immediately
            del crosscoder_input
            th.cuda.empty_cache()
        
        # Create joke data
        joke_data = {
            'joke_index': joke_idx,
            'joke_text': joke,
            'nonzero_feature_count': len(nonzero_feature_indices),
            'nonzero_features': nonzero_feature_indices,
            'kl_divergence': kl_div
        }
        
        # Process feature metrics one by one to save memory
        for feat_idx in nonzero_feature_indices:
            # Access just the needed decoder vectors
            with th.no_grad():
                base_vec = crosscoder.decoder.weight[0, feat_idx].detach().clone()
                it_vec = crosscoder.decoder.weight[1, feat_idx].detach().clone()
                
                # Calculate metrics
                l1_norm_base = th.norm(base_vec, p=1).item()
                l1_norm_it = th.norm(it_vec, p=1).item()
                l2_norm_base = th.norm(base_vec, p=2).item()
                l2_norm_it = th.norm(it_vec, p=2).item()
                cosine = cos_sim(base_vec, it_vec).item()
                
                # Store metrics
                feature_data.append({
                    'joke_index': joke_idx,
                    'feature_index': feat_idx,
                    'l1_norm_base': l1_norm_base,
                    'l1_norm_it': l1_norm_it,
                    'l2_norm_base': l2_norm_base,
                    'l2_norm_it': l2_norm_it,
                    'cosine_similarity': cosine
                })
                
                # Clean up
                del base_vec, it_vec
        
        # Final cleanup
        del features, nonzero_mask, nonzero_idx
        th.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error processing joke {joke_idx}: {e}")
        joke_data = {
            'joke_index': joke_idx,
            'joke_text': joke,
            'nonzero_feature_count': 0,
            'nonzero_features': [],
            'kl_divergence': None
        }
    
    return joke_data, feature_data

def calculate_kl_divergence(logits_model1, logits_model2, temperature=1.0):
    """
    Calculate KL divergence between two sets of logits.
    """
    # Apply temperature scaling
    if temperature != 1.0:
        logits_model1 = logits_model1 / temperature
        logits_model2 = logits_model2 / temperature
    
    # Convert to log probabilities and probabilities
    log_probs_model1 = F.log_softmax(logits_model1, dim=-1)
    probs_model1 = F.softmax(logits_model1, dim=-1)
    
    # Use PyTorch's KL divergence implementation
    kl_div = F.kl_div(
        F.log_softmax(logits_model2, dim=-1),
        probs_model1,
        reduction='batchmean',
        log_target=False
    )
    
    return kl_div

def collect_with_cpu_dataframes(jokes, gemma_2, gemma_2_it, crosscoder, token_index=-5, save_dir='saved_data'):
    """
    Memory-efficient data collection that creates CPU-based DataFrames.
    
    Args:
        jokes: List of joke strings
        gemma_2: Base model
        gemma_2_it: Instruction-tuned model
        crosscoder: Crosscoder model
        token_index: Token position to analyze (default: -5)
        save_dir: Directory to save the collected data
    
    Returns:
        Tuple of (jokes_df, features_df, global_df)
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize lists to collect data
    all_joke_data = []
    all_feature_data = []
    
    # Process each joke individually
    for idx, joke in enumerate(tqdm(jokes, desc="Processing jokes")):
        # Process the joke
        joke_data, feature_data = process_single_joke(
            joke, idx, gemma_2, gemma_2_it, crosscoder, token_index
        )
        
        # Add to our lists
        all_joke_data.append(joke_data)
        all_feature_data.extend(feature_data)
        
        # Save incremental results every 10 jokes
        if (idx + 1) % 10 == 0 or idx == len(jokes) - 1:
            # Create temporary DataFrames just for saving
            temp_jokes_df = pd.DataFrame(all_joke_data)
            temp_features_df = pd.DataFrame(all_feature_data)
            
            # Save intermediate results
            temp_jokes_df.to_csv(os.path.join(save_dir, f'jokes_metrics_partial_{idx+1}.csv'), index=False)
            temp_features_df.to_csv(os.path.join(save_dir, f'feature_metrics_partial_{idx+1}.csv'), index=False)
            
            print(f"Saved intermediate results after processing {idx+1} jokes")
            print(f"Current memory usage: {len(all_joke_data)} joke entries, {len(all_feature_data)} feature entries")
            
            # Clear memory for temporary DataFrames
            del temp_jokes_df, temp_features_df
            gc.collect()
        
        # Force cleanup
        gc.collect()
        th.cuda.empty_cache()
    
    # Create final CPU-based DataFrames
    print("Creating final DataFrames on CPU...")
    jokes_df = pd.DataFrame(all_joke_data)
    features_df = pd.DataFrame(all_feature_data)
    
    # Save final results
    jokes_df.to_csv(os.path.join(save_dir, 'jokes_metrics.csv'), index=False)
    features_df.to_csv(os.path.join(save_dir, 'feature_metrics.csv'), index=False)
    
    # Compute global decoder stats
    print("Computing global decoder stats...")
    global_df = compute_global_decoder_stats(crosscoder, save_dir)
    
    return jokes_df, features_df, global_df

def compute_global_decoder_stats(crosscoder, save_dir):
    """
    Compute global statistics for all decoder directions, creating a CPU DataFrame.
    
    Args:
        crosscoder: Crosscoder model
        save_dir: Directory to save the stats
    
    Returns:
        DataFrame with global decoder stats
    """
    # Get total number of features
    num_features = crosscoder.decoder.weight.shape[1]
    print(f"Computing global stats for {num_features} features...")
    
    # Process features in batches to save memory
    batch_size = 1000
    all_stats = []
    
    # Setup cosine similarity function
    cos_sim = nn.CosineSimilarity(dim=0)
    
    for batch_start in tqdm(range(0, num_features, batch_size), desc="Computing global stats", 
                          total=(num_features + batch_size - 1) // batch_size):
        batch_end = min(batch_start + batch_size, num_features)
        
        batch_stats = []
        for feat_idx in range(batch_start, batch_end):
            with th.no_grad():
                # Get decoder vectors for this feature
                base_vec = crosscoder.decoder.weight[0, feat_idx].detach().cpu()
                it_vec = crosscoder.decoder.weight[1, feat_idx].detach().cpu()
                
                # Calculate metrics
                l1_norm_base = th.norm(base_vec, p=1).item()
                l1_norm_it = th.norm(it_vec, p=1).item()
                l2_norm_base = th.norm(base_vec, p=2).item()
                l2_norm_it = th.norm(it_vec, p=2).item()
                cosine = cos_sim(base_vec, it_vec).item()
                
                # Calculate ratios
                l1_ratio = l1_norm_it / l1_norm_base if l1_norm_base > 0 else float('inf')
                l2_ratio = l2_norm_it / l2_norm_base if l2_norm_base > 0 else float('inf')
                
                # Calculate delta norm
                delta_norm_val = 0.5 * (1 + (l2_norm_it - l2_norm_base) / max(l2_norm_it, l2_norm_base)) if max(l2_norm_it, l2_norm_base) > 0 else 0.5
                
                # Store stats
                batch_stats.append({
                    'feature_index': feat_idx,
                    'l1_norm_base': l1_norm_base,
                    'l1_norm_it': l1_norm_it,
                    'l2_norm_base': l2_norm_base,
                    'l2_norm_it': l2_norm_it,
                    'cosine_similarity': cosine,
                    'norm_ratio_l1': l1_ratio,
                    'norm_ratio_l2': l2_ratio,
                    'delta_norm': delta_norm_val
                })
        
        # Add batch stats to overall stats
        all_stats.extend(batch_stats)
        
        # Save batch progress
        temp_df = pd.DataFrame(batch_stats)
        temp_df.to_csv(os.path.join(save_dir, f'global_stats_batch_{batch_start}_{batch_end-1}.csv'), index=False)
        
        # Clear memory
        del batch_stats, temp_df
        gc.collect()
        th.cuda.empty_cache()
    
    # Create final CPU DataFrame
    global_df = pd.DataFrame(all_stats)
    
    # Save final results
    global_df.to_csv(os.path.join(save_dir, 'global_decoder_stats.csv'), index=False)
    
    return global_df

def find_interesting_directions(global_df, cosine_threshold=0.8, norm_threshold=0.1):
    """
    Helper function to find interesting decoder directions with:
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