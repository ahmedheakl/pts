

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json
import os
from scipy import stats
from scipy import stats as scipy_stats
from datetime import datetime

# Helper function to analyze results across all workers
def analyze_attention_results(all_results, architecture, model_diffusion, model_llm, benchmark_name):
    """
    Analyze attention patterns across all workers' results
    
    Args:
        all_results: Combined results from all workers (list of result dicts)
    """
    # Extract samples with attention metrics
    attention_samples = [r for r in all_results if 'attention_metrics' in r and r['attention_metrics']]
    
    if not attention_samples:
        print("No attention metrics found in results")
        return
    
    # Extract metrics
    plan_attention_ratios = [r['attention_metrics']['plan_attention_ratio'] for r in attention_samples]
    question_attention_ratios = [r['attention_metrics']['question_attention_ratio'] for r in attention_samples]
    plan_to_question_ratios = [r['attention_metrics']['plan_to_question_ratio'] for r in attention_samples] #if less than 1 then more attention to question
    attention_entropies = [r['attention_metrics']['attention_entropy'] for r in attention_samples]
    
    # Analyze correlation with correctness
    correct_samples = [r for r in attention_samples if r['is_correct']]
    incorrect_samples = [r for r in attention_samples if not r['is_correct']]

    # Statistical significance test
    t_stat, p_value = scipy_stats.ttest_ind(
        [r['attention_metrics']['plan_attention_ratio'] for r in correct_samples] if correct_samples else [0],
        [r['attention_metrics']['plan_attention_ratio'] for r in incorrect_samples] if incorrect_samples else [0]
    )

    # Distribution analysis
    high_plan_attention = sum(1 for ratio in plan_attention_ratios if ratio > 0.3)
    plan_focused = sum(1 for ratio in plan_to_question_ratios if ratio > 1.0)

    stats = {
        "architecture": architecture,
        "model_diffusion": model_diffusion,
        "model_llm": model_llm,
        "num_samples": len(attention_samples),
        "plan_attention_ratio": {
            "mean": float(np.mean(plan_attention_ratios)),
            "std": float(np.std(plan_attention_ratios))
        },
        "question_attention_ratio": {
            "mean": float(np.mean(question_attention_ratios)),
            "std": float(np.std(question_attention_ratios))
        },
        "plan_to_question_ratio": {
            "mean": float(np.mean(plan_to_question_ratios)),
            "std": float(np.std(plan_to_question_ratios))
        },
        "attention_entropy": {
            "mean": float(np.mean(attention_entropies)),
            "std": float(np.std(attention_entropies))
        },
        "t_stat": float(t_stat) if t_stat is not None else None,
        "p_value": float(p_value) if p_value is not None else None,
        "high_plan_attention_count": high_plan_attention,
        "high_plan_attention_percent": 100 * high_plan_attention / len(attention_samples) if attention_samples else 0.0,
        "plan_focused_count": plan_focused,
        "plan_focused_percent": 100 * plan_focused / len(attention_samples) if attention_samples else 0.0
    }

    
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), f'../../outputs/attention')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{benchmark_name}_attention_{time_stamp}.json')
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n=== OVERALL ATTENTION ANALYSIS ({len(attention_samples)} samples) ===")
    print(f"Plan Attention Ratio - Mean: {stats['plan_attention_ratio']['mean']:.4f}, Std: {stats['plan_attention_ratio']['std']:.4f}")
    print(f"Question Attention Ratio - Mean: {stats['question_attention_ratio']['mean']:.4f}, Std: {stats['question_attention_ratio']['std']:.4f}")
    print(f"Plan-to-Question Ratio - Mean: {stats['plan_to_question_ratio']['mean']:.4f}, Std: {stats['plan_to_question_ratio']['std']:.4f}")
    print(f"Attention Entropy - Mean: {stats['attention_entropy']['mean']:.4f}, Std: {stats['attention_entropy']['std']:.4f}")
    
    # Attention vs. correctness analysis
    if correct_samples and incorrect_samples:
        correct_plan_attention = np.mean([r['attention_metrics']['plan_attention_ratio'] for r in correct_samples])
        incorrect_plan_attention = np.mean([r['attention_metrics']['plan_attention_ratio'] for r in incorrect_samples])
        
        print(f"\n=== ATTENTION vs. CORRECTNESS ===")
        print(f"Correct answers - Average plan attention: {correct_plan_attention:.4f} ({len(correct_samples)} samples)")
        print(f"Incorrect answers - Average plan attention: {incorrect_plan_attention:.4f} ({len(incorrect_samples)} samples)")
        print(f"Difference: {correct_plan_attention - incorrect_plan_attention:.4f}")
        
        # Statistical significance test
        correct_attentions = [r['attention_metrics']['plan_attention_ratio'] for r in correct_samples]
        incorrect_attentions = [r['attention_metrics']['plan_attention_ratio'] for r in incorrect_samples]
        
    t_stat, p_value = scipy_stats.ttest_ind(correct_attentions, incorrect_attentions)
    print(f"T-test p-value: {p_value:.6f} {'(significant)' if p_value < 0.05 else '(not significant)'}")
    
    # Distribution analysis
    high_plan_attention = sum(1 for ratio in plan_attention_ratios if ratio > 0.3)
    print(f"\nSamples with high plan attention (>30%): {high_plan_attention}/{len(attention_samples)} ({100*high_plan_attention/len(attention_samples):.1f}%)")
    
    plan_focused = sum(1 for ratio in plan_to_question_ratios if ratio > 1.0)
    print(f"Samples where plan gets more attention than question: {plan_focused}/{len(attention_samples)} ({100*plan_focused/len(attention_samples):.1f}%)")





def analyze_plan_attention(model, tokenizer, question: str, plan: str, template: str = None) -> Dict:
    """
    Function to analyze how much attention the LLM pays to the plan vs question.
    
    Args:
        model: The loaded LLM model (should have output_attentions=True capability)
        tokenizer: The tokenizer for the model
        question: The original question
        plan: The diffusion-generated plan
        template: Template for formatting input (should match your LLM_TEMPLATE)
    
    Returns:
        Dictionary with attention metrics
    """
    if template is None:
        template = "Question: {question}\nPlan: {plan}\nAnswer:"
    
    # Format input (use your existing template)
    formatted_input = template.format(question=question, plan=plan)
    
    # Tokenize
    inputs = tokenizer(formatted_input, return_tensors="pt", return_offsets_mapping=True)
    input_ids = inputs['input_ids'].to(model.device)
    offsets = inputs['offset_mapping'][0] if 'offset_mapping' in inputs else None
    offsets= offsets[2:]


    # Identify plan and question token positions
    plan_tokens, question_tokens = _identify_token_regions(formatted_input, question, plan, offsets)
    
    # Extract attention weights
    with torch.no_grad():
        # Set model to output attentions if not already configured: basic boolean flag
        original_output_attentions = getattr(model.config, 'output_attentions', False)
        model.config.output_attentions = True
        
        try:
            outputs = model(input_ids, output_attentions=True)
            attentions = outputs.attentions
        finally:
            # Restore original setting
            model.config.output_attentions = original_output_attentions
    
    # Compute attention metrics
    metrics = _compute_attention_metrics(attentions, plan_tokens, question_tokens)
    
    return metrics


def _identify_token_regions(formatted_input: str, question: str, plan: str, offsets) -> Tuple[List[int], List[int]]:
    """Identify which tokens belong to plan vs question regions."""
    plan_tokens = []
    question_tokens = []
    
    if offsets is None:
        # Fallback: rough estimation based on string positions
        question_start = formatted_input.find(question)
        question_end = question_start + len(question) if question_start != -1 else 0
        plan_start = formatted_input.find(plan)
        plan_end = plan_start + len(plan) if plan_start != -1 else 0
        
        # This is a rough approximation -
        total_length = len(formatted_input)
        if question_start != -1: #if found
            question_ratio_start = question_start / total_length
            question_ratio_end = question_end / total_length
        if plan_start != -1: #if found
            plan_ratio_start = plan_start / total_length
            plan_ratio_end = plan_end / total_length
        
        # Rough token estimation (this is approximate)
        seq_len = len(formatted_input.split())  # Very rough token count, we suppose that number of words ~ tokens
        if question_start != -1:
            question_tokens = list(range(int(question_ratio_start * seq_len), int(question_ratio_end * seq_len)))
            #list of question token indices
        if plan_start != -1:
            plan_tokens = list(range(int(plan_ratio_start * seq_len), int(plan_ratio_end * seq_len)))
            #list of plan token indices
    else:
        # More accurate version using offset mappings
        question_start = formatted_input.find(question)
        question_end = question_start + len(question) if question_start != -1 else 0
        plan_start = formatted_input.find(plan)
        plan_end = plan_start + len(plan) if plan_start != -1 else 0
        
        for i, (start_char, end_char) in enumerate(offsets):
            if start_char is None or end_char is None:
                continue
                
            # Check if token overlaps with question
            if question_start != -1 and start_char < question_end and end_char > question_start:
                question_tokens.append(i)
            
            # Check if token overlaps with plan
            if plan_start != -1 and start_char < plan_end and end_char > plan_start:
                plan_tokens.append(i)
    
    return plan_tokens, question_tokens


def _compute_attention_metrics(attentions: List[torch.Tensor], 
                              plan_tokens: List[int], 
                              question_tokens: List[int]) -> Dict:
    """Compute attention metrics from attention tensors."""
    
    if not attentions or len(plan_tokens) == 0:
        return {
            'plan_attention_ratio': 0.0,
            'question_attention_ratio': 0.0,
            'plan_to_question_ratio': 0.0,
            'attention_entropy': 0.0,
            'layer_wise_plan_attention': []
        }
    
    num_layers = len(attentions)
    seq_len = attentions[0].shape[-1]
    
    # Aggregate attention across layers and heads
    total_attention = torch.zeros(seq_len, seq_len, device=attentions[0].device)
    layer_wise_plan_attention = []
    
    for attention_layer in attentions:
        # attention_layer shape: [batch=1, num_heads, seq_len, seq_len]
        layer_avg = attention_layer.squeeze(0).mean(dim=0)  # Average across heads
        total_attention += layer_avg
        
        # Calculate plan attention for this layer
        if plan_tokens:
            layer_plan_attention = layer_avg[:, plan_tokens].sum().item()
            layer_wise_plan_attention.append(layer_plan_attention)
        else:
            layer_wise_plan_attention.append(0.0)
    
    # Average across layers
    avg_attention = total_attention / num_layers
    
    # Calculate attention sums
    total_attention_sum = avg_attention.sum().item()
    plan_attention_sum = avg_attention[:, plan_tokens].sum().item() if plan_tokens else 0.0
    question_attention_sum = avg_attention[:, question_tokens].sum().item() if question_tokens else 0.0
    
    # Calculate ratios
    plan_attention_ratio = plan_attention_sum / total_attention_sum if total_attention_sum > 0 else 0.0
    question_attention_ratio = question_attention_sum / total_attention_sum if total_attention_sum > 0 else 0.0
    

    #ALSSO !!! Normalize by the number of tokens in each region to get per-token attention because nb of tokens can vary (for plan and question)
    if plan_tokens:
        plan_attention_ratio /= len(plan_tokens)
    if question_tokens:
        question_attention_ratio /= len(question_tokens)

    plan_to_question_ratio = plan_attention_sum / question_attention_sum if question_attention_sum > 0 else 0.0


    # Calculate attention entropy
    attention_probs = avg_attention.sum(dim=0)  # Sum over query positions
    attention_probs = attention_probs / attention_probs.sum()  # Normalize
    attention_entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-12)).item() #1e-12 to avoid log(0)
    
    return {
        'plan_attention_ratio': plan_attention_ratio,
        'question_attention_ratio': question_attention_ratio,
        'plan_to_question_ratio': plan_to_question_ratio,
        'attention_entropy': attention_entropy,
        'layer_wise_plan_attention': layer_wise_plan_attention,
        'num_plan_tokens': len(plan_tokens),
        'num_question_tokens': len(question_tokens)
    }
