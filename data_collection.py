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

def process_single_joke(joke, joke_idx, gemma_2, gemma_2_it, crosscoder, token_index_range=list(range(-10,0,1))):
    """
    Process a single joke with minimal memory usage.
    
    Args:
        joke: The joke string
        joke_idx: Index of the joke
        gemma_2: Base model
        gemma_2_it: Instruction-tuned model
        crosscoder: Crosscoder model
        token_index_range: Token positions to analyze
    
    Returns:
        Tuple of (joke_data, feature_data)
    """
    # Setup cosine similarity function
    cos_sim = nn.CosineSimilarity(dim=0)
    
    # Initialize containers
    joke_data = {}

    try:
        # --- First get activations only ---
        with th.inference_mode():
            with gemma_2.trace(joke):
                l13_act_base = gemma_2.model.layers[13].output[0][:, token_index_range].save()
                gemma_2.model.layers[13].output.stop()
                logits_base = gemma_2.model.output[0][:, token_index_range].save()
                gemma_2.model.output.stop()
            
            with gemma_2_it.trace(joke):
                l13_act_it = gemma_2_it.model.layers[13].output[0][:, token_index_range].save()
                gemma_2_it.model.layers[13].output.stop()        
                logits_it = gemma_2_it.model.output[0][:, token_index_range].save()
                gemma_2_it.model.output.stop()
        
        # Calculate KL divergence immediately
        kl_div = calculate_kl_divergence(logits_base, logits_it).item()
        
        # Clean up logits to save memory
        del logits_base, logits_it
        
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
        
        # Create joke data
        joke_data = {
            'joke_index': joke_idx,
            'joke_text': joke,
            'nonzero_feature_count': len(nonzero_feature_indices),
            'nonzero_features': nonzero_feature_indices,
            'kl_divergence': kl_div
        }
              
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
    
    return joke_data

def calculate_kl_divergence(logits_model1, logits_model2, temperature=1.0):
    """
    Calculate KL divergence between two sets of logits.
    
    Args:
        logits_model1: Tensor of shape [batch_size, num_tokens, vocab_size] or [num_tokens, vocab_size]
        logits_model2: Tensor of shape [batch_size, num_tokens, vocab_size] or [num_tokens, vocab_size]
        temperature: Temperature scaling parameter
    
    Returns:
        Tensor of shape [batch_size, num_tokens] or [num_tokens] containing KL divergence per token
    """
    # Apply temperature scaling
    if temperature != 1.0:
        logits_model1 = logits_model1 / temperature
        logits_model2 = logits_model2 / temperature
    
    # Convert to probabilities
    probs_model1 = F.softmax(logits_model1, dim=-1)
    log_probs_model2 = F.log_softmax(logits_model2, dim=-1)
    
    # Calculate KL divergence for each token separately
    # KL(P||Q) = sum(P(x) * log(P(x)/Q(x))) = sum(P(x) * (log P(x) - log Q(x)))
    kl_per_token = (probs_model1 * (F.log_softmax(logits_model1, dim=-1) - log_probs_model2)).sum(dim=-1)
    
    return kl_per_token


def collect_data(jokes, gemma_2, gemma_2_it, crosscoder, token_index_range=list(range(-10,0,1)), save_dir='saved_data', low_mem=False):
    """
    Memory-efficient data collection that creates CPU-based DataFrames.
    
    Args:
        jokes: List of joke strings
        gemma_2: Base model
        gemma_2_it: Instruction-tuned model
        crosscoder: Crosscoder model
        token_index_range: Token positions to analyze (default: -10:-1)
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
        joke_data = process_single_joke(joke, idx, gemma_2, gemma_2_it, crosscoder, token_index_range, low_mem)
        
        # Add to our lists
        all_joke_data.append(joke_data)
        
        # # Save incremental results every 10 jokes
        # if (idx + 1) % 10 == 0 or idx == len(jokes) - 1:
        #     # Create temporary DataFrames just for saving
        #     temp_jokes_df = pd.DataFrame(all_joke_data)
        #     temp_features_df = pd.DataFrame(all_feature_data)
            
        #     # Save intermediate results
        #     temp_jokes_df.to_csv(os.path.join(save_dir, f'jokes_metrics_partial_{idx+1}.csv'), index=False)
        #     temp_features_df.to_csv(os.path.join(save_dir, f'feature_metrics_partial_{idx+1}.csv'), index=False)
            
        #     print(f"Saved intermediate results after processing {idx+1} jokes")
        #     print(f"Current memory usage: {len(all_joke_data)} joke entries, {len(all_feature_data)} feature entries")
            
        #     # Clear memory for temporary DataFrames
        #     del temp_jokes_df, temp_features_df
        #     gc.collect()
        
        # Force cleanup
        gc.collect()
        th.cuda.empty_cache()
    
    # Create final CPU-based DataFrames
    print("Creating final DataFrames on CPU...")
    jokes_df = pd.DataFrame(all_joke_data)    
    # Save final results
    jokes_df.to_csv(os.path.join(save_dir, 'jokes_metrics.csv'), index=False)
    
    # Compute global decoder stats
    print("Computing global decoder stats...")
    global_df = compute_global_decoder_stats(crosscoder, save_dir)
    
    return jokes_df, global_df

def compute_global_decoder_stats(crosscoder, save_dir, low_mem=False):
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
    if low_mem:
        # Process feature metrics one by one to save memory
        for feat_idx in range(num_features):
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
                
                # Calculate delta norms
                delta_l1_norm = abs(l1_norm_base - l1_norm_it)
                delta_l2_norm = abs(l2_norm_base - l2_norm_it)
                
                cosine = cos_sim(base_vec, it_vec).item()
                
                # Store metrics
                all_stats.append({
                    'feature_index': feat_idx,
                    'l1_norm_base': l1_norm_base,
                    'l1_norm_it': l1_norm_it,
                    'l2_norm_base': l2_norm_base,
                    'l2_norm_it': l2_norm_it,
                    'delta_l1_norm': delta_l1_norm,
                    'delta_l2_norm': delta_l2_norm,
                    'cosine_similarity': cosine
                })                    
                # Clean up
                del base_vec, it_vec
        # Create final CPU DataFrame
        global_df = pd.DataFrame(all_stats)
        

    else:
        # Process all features at once more efficiently
        with th.no_grad():
            # Get all decoder vectors for nonzero features at once
            base_vecs = crosscoder.decoder.weight[0].detach()
            it_vecs = crosscoder.decoder.weight[1].detach()
            
            # Calculate metrics for all vectors at once
            l1_norms_base = th.norm(base_vecs, p=1, dim=1)
            l1_norms_it = th.norm(it_vecs, p=1, dim=1)
            l2_norms_base = th.norm(base_vecs, p=2, dim=1)
            l2_norms_it = th.norm(it_vecs, p=2, dim=1)
            
            # Calculate delta norms (difference between base and IT model norms)
            delta_l1_norms = th.abs(l1_norms_base - l1_norms_it)
            delta_l2_norms = th.abs(l2_norms_base - l2_norms_it)
            
            # Calculate cosine similarities
            cosines = th.nn.functional.cosine_similarity(base_vecs, it_vecs, dim=1)
            
            # Convert to lists for DataFrame creation
            global_df = pd.DataFrame({'feat_idx' : list(range(num_features)),
                                      'l1_norm_base': l1_norm_base.cpu().numpy(),
                                      'l1_norm_it': l1_norm_it.cpu().numpy(),
                                      'l2_norm_base': l2_norm_base.cpu().numpy(),
                                      'l2_norm_it': l2_norm_it.cpu().numpy(),
                                      'delta_l1_norm': delta_l1.cpu().numpy(),
                                      'delta_l2_norm': delta_l2.cpu().numpy(),
                                      'cosine_similarity': cosine.cpu().numpy()})
    
    
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