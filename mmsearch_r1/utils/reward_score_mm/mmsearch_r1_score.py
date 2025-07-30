# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import math
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import wandb

# Initialize SBERT model
try:
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Warning: Could not load SBERT model: {e}")
    sbert_model = None

# Global counter for reward calls
reward_call_counts = {
    "bm25": 0,
    "f1": 0,
    "recall": 0,
    "precision": 0,
    "sbert_cosine": 0
}

# Running statistics for reward normalization
reward_stats = {
    'em': {'mean': 0.0, 'std': 1.0, 'count': 0},
    'bm25': {'mean': 0.0, 'std': 1.0, 'count': 0},
    'f1': {'mean': 0.0, 'std': 1.0, 'count': 0},
    'recall': {'mean': 0.0, 'std': 1.0, 'count': 0},
    'precision': {'mean': 0.0, 'std': 1.0, 'count': 0},
    'sbert': {'mean': 0.0, 'std': 1.0, 'count': 0}
}

# Previous state potentials for shaping
previous_states = {}

# adapted from search-r1
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    '''
    prediction: string
    golden_answers: list or string, support multi candidate answers
    '''
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    exactly_match = False
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            exactly_match = True
            break
    return exactly_match


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(prediction):
    """Extract the answer from the solution string."""
    # 如果输入是列表，取最后一个元素
    if isinstance(prediction, list):
        prediction = prediction[-1]
    
    # 确保prediction是字符串
    if not isinstance(prediction, str):
        print(f"[Warning] Prediction is not a string: {type(prediction)}")
        return None
    
    # 尝试提取<answer>标签中的内容
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, prediction, re.DOTALL)
    matches = list(match)
    
    if matches:
        return matches[-1].group(1).strip()
    
    # 如果没有找到<answer>标签，尝试提取最后一个完整的句子
    sentences = re.split(r'[.!?]', prediction)
    if sentences:
        last_sentence = sentences[-1].strip()
        if last_sentence:
            print("[Warning] No answer tag found, using last sentence as answer:", last_sentence)
            return last_sentence
    
    print("[Warning] No valid answer found in prediction")
    return None


def is_valid_direct_answer(response, direct_answer_format) -> bool:
    """
    Check Direct Answer: <reason>...</reason><answer>...</answer>
      1) Structure Matching
      2) Pattern Count: <reason>...</reason>, <answer>...</answer>
      3) No any search actions included
    """
    pattern = direct_answer_format
    # 1). Structure Matching
    if not re.match(pattern, response, re.DOTALL):
        return False
    # 2). Pattern Count
    if response.count('<reason>') != 1 or response.count('</reason>') != 1:
        return False
    if response.count('<answer>') != 1 or response.count('</answer>') != 1:
        return False
    # 3). <search><img></search> or <text_search> is not allowed!
    if '<search><img></search>' in response:
        return False
    if '<text_search>' in response or '</text_search>' in response:
        return False
    return True


def is_valid_image_search(response, call_image_search_format) -> bool:
    """
    Check Image Search: <reason>...</reason>...<search><img></search>
      1) Structure Matching
      2) Pattern Count: <reason>...</reason>
      3) Pattern Count: <search><img></search>
      4) No <answer> or </answer> or <text_search> or </text_search> included
    """
    pattern = call_image_search_format
    # 1). Structure Matching
    if not re.match(pattern, response, re.DOTALL):
        return False
    # 2). <reason> Count
    if response.count('<reason>') != 1 or response.count('</reason>') != 1:
        return False
    # 3). <search><img></search> Count
    if response.count('<search><img></search>') != 1:
        return False
    # 4). <answer> or <text_search> is not allowed!
    if '<answer>' in response or '</answer>' in response:
        return False
    if '<text_search>' in response or '</text_search>' in response:
        return False
    return True


def is_valid_text_search(response, call_text_search_format) -> bool:
    """
    Check Text Search: <reason>...</reason>...<text_search>...</text_search>
      1) Structure Matching
      2) Pattern Count: <reason>...</reason> 
      3) Pattern Count: <text_search>...</text_search> 
      4) No <answer> or </answer> or <search><img></search> included
    """
    pattern = call_text_search_format
    # 1). Structure Matching
    if not re.match(pattern, response, re.DOTALL):
        return False
    # 2). <reason> Count
    if response.count('<reason>') != 1 or response.count('</reason>') != 1:
        return False
    # 3). <text_search> and </text_search> Count
    if response.count('<text_search>') != 1 or response.count('</text_search>') != 1:
        return False
    # 4). <answer> or <search><img></search> is not allowed!
    if '<answer>' in response or '</answer>' in response:
        return False
    if '<search><img></search>' in response:
        return False
    return True


def format_reward(input_string: list):
    """
    Check if the model's response follows the required formats and return a reward.
    [1-turn]:
        - Direct Answer
    [2-turn]:
        - Call Image Search + Answer
        - Call Text Search + Answer
    [3-turn]:
        - Call Image Search + Call Text Search + Answer
    Args:
    - input_string (list): A list of responses, currently, max length of `input_string` is 3 (3-turn).
    Returns:
    - format_score: float, 1.0 for right format, 0.0 for wrong
    - search_count: int, times of search tools called
    """
    conv_rounds = len(input_string)
    format_score, search_count = 0, 0
    # All allowed formats
    direct_answer_format = r'^<reason>.*</reason>.*<answer>.*</answer>$'
    call_image_search_format = r'^<reason>.*</reason>.*<search><img></search>$'
    call_text_search_format = r'^<reason>.*</reason>.*<text_search>.*</text_search>$'
    # HACK/FIXME: We need more flexible judge in the future
    # 1-turn
    if conv_rounds == 1:
        response_1 = input_string[0].strip()
        if (
            ("<search><img></search>" in response_1)
            or ("<text_search>" in response_1 and "</text_search>" in response_1)
        ):
            search_count += 1
        # Direct Answer
        if is_valid_direct_answer(response_1, direct_answer_format):
            format_score = 1
    # 2-turn
    elif conv_rounds == 2:
        response_1, response_2 = input_string[0].strip(), input_string[1].strip()
        if (
            ("<search><img></search>" in response_1)
            or ("<text_search>" in response_1 and "</text_search>" in response_1)
        ):
            search_count += 1
        # Call Image Search + Answer
        if is_valid_image_search(response_1, call_image_search_format) and is_valid_direct_answer(response_2, direct_answer_format):
            format_score = 1
        # Call Text Search + Answer
        elif is_valid_text_search(response_1, call_text_search_format) and is_valid_direct_answer(
            response_2, direct_answer_format
        ):
            format_score = 1
    # 3-turn
    elif conv_rounds == 3:
        response_1, response_2, response_3 = input_string[0].strip(), input_string[1].strip(), input_string[2].strip()
        if (
            ("<search><img></search>" in response_1)
            or ("<text_search>" in response_1 and "</text_search>" in response_1)
        ):
            search_count += 1
        if (
            ("<search><img></search>" in response_2)
            or ("<text_search>" in response_2 and "</text_search>" in response_2)
        ):
            search_count += 1
        # Call Image Search + Call Text Search + Answer
        if (
            is_valid_image_search(response_1, call_image_search_format)
            and is_valid_text_search(response_2, call_text_search_format)
            and is_valid_direct_answer(response_3, direct_answer_format)
        ):
            format_score = 1
    else:
        raise ValueError(f"[Error Occured] Number of responses is {conv_rounds}, which is not supported currently!")

    return format_score, search_count


def compute_bm25_score(prediction, ground_truth):
    """计算BM25分数"""
    # 创建语料库
    corpus = [ground_truth, prediction]
    
    # BM25参数
    k1 = 1.2
    b = 0.75
    
    # 计算平均文档长度
    N = len(corpus)
    avgdl = sum(len(doc.split()) for doc in corpus) / N if N > 0 else 0
    
    # 计算IDF
    def idf(term):
        n_t = sum(1 for doc in corpus if term in doc.split())
        return math.log((N - n_t + 0.5) / (n_t + 0.5) + 1) if n_t > 0 else 0
    
    # 使用ground_truth作为查询，prediction作为文档计算BM25分数
    query_terms = ground_truth.split()
    if not query_terms:  # Handle empty query terms
        return 0.0
        
    doc_terms = prediction.split()
    doc_len = len(doc_terms)
    
    bm25_score = 0
    for term in query_terms:
        f_td = doc_terms.count(term)  # 词频
        idf_t = idf(term)
        if f_td > 0 and idf_t > 0:
            numerator = f_td * (k1 + 1)
            denominator = f_td + k1 * (1 - b + b * (doc_len / avgdl)) if avgdl > 0 else 1
            bm25_score += idf_t * (numerator / denominator)  # 累加每个词的分数
    
    # 归一化BM25分数到0-1范围
    return min(bm25_score / (len(query_terms) * 2.0), 1.0)  # 根据查询词数量归一化

def compute_f1_score(prediction, ground_truth):
    """计算F1分数"""
    # 分词
    pred_tokens = set(prediction.lower().split())
    truth_tokens = set(ground_truth.lower().split())
    
    if not truth_tokens or not pred_tokens:
        return 0.0
    
    # 计算交集
    common_tokens = pred_tokens.intersection(truth_tokens)
    
    # 计算precision和recall
    precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
    recall = len(common_tokens) / len(truth_tokens) if truth_tokens else 0
    
    # 计算F1
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    return f1

def compute_recall_score(prediction, ground_truth):
    """计算召回率"""
    pred_tokens = set(prediction.lower().split())
    truth_tokens = set(ground_truth.lower().split())
    
    if not truth_tokens:
        return 0.0
    
    common_tokens = pred_tokens.intersection(truth_tokens)
    recall = len(common_tokens) / len(truth_tokens)
    
    return recall

def compute_precision_score(prediction, ground_truth):
    """计算精确率"""
    pred_tokens = set(prediction.lower().split())
    truth_tokens = set(ground_truth.lower().split())
    
    if not pred_tokens:
        return 0.0
    
    common_tokens = pred_tokens.intersection(truth_tokens)
    precision = len(common_tokens) / len(pred_tokens)
    
    return precision

def compute_sbert_cosine_score(prediction, ground_truth):
    """计算SBERT余弦相似度"""
    if sbert_model is None:
        return 0.0
    
    try:
        # 生成embeddings
        pred_embedding = sbert_model.encode([prediction], convert_to_tensor=True)
        truth_embedding = sbert_model.encode([ground_truth], convert_to_tensor=True)
        
        # 转换为numpy数组
        if torch.is_tensor(pred_embedding):
            pred_embedding = pred_embedding.cpu().numpy()
        if torch.is_tensor(truth_embedding):
            truth_embedding = truth_embedding.cpu().numpy()
        
        # 计算余弦相似度
        similarity = cosine_similarity(pred_embedding, truth_embedding)[0][0]
        return float(similarity)
    except Exception as e:
        print(f"Error in SBERT calculation: {e}")
        return 0.0

def update_running_stats(reward_type, value):
    """Update running mean and standard deviation estimates"""
    stats = reward_stats[reward_type]
    stats['count'] += 1
    delta = value - stats['mean']
    stats['mean'] += delta / stats['count']
    delta2 = value - stats['mean']
    stats['std'] = math.sqrt(((stats['count'] - 1) * (stats['std'] ** 2) + delta * delta2) / stats['count'])

def normalize_reward(reward_type, value):
    """
    Normalize reward using running statistics and ensure non-negative output
    Maps to [0,1] range after normalization
    """
    stats = reward_stats[reward_type]
    if stats['count'] < 2:  # Not enough samples for normalization
        return value
    # First normalize
    normalized = (value - stats['mean']) / (stats['std'] + 1e-5)
    # Then map to [0,1] using sigmoid
    return 1 / (1 + math.exp(-normalized))

def compute_fuzzy_reward(reward_type, value):
    """
    Compute fuzzy reward component using membership functions
    This creates a smoother reward signal compared to exact rewards
    Ensures output is in [0,1]
    """
    if reward_type == 'em':
        # For exact match, use a step function
        return float(value > 0.5)
    elif reward_type in ['bm25', 'f1', 'recall', 'precision', 'sbert']:
        # For continuous rewards, use a smooth sigmoid function
        # Already ensures output in [0,1]
        return 1 / (1 + math.exp(-10 * (value - 0.5)))
    return min(max(value, 0), 1)  # Clamp to [0,1]

def compute_potential(state_id, reward_scores):
    """
    Compute potential function φ(s) using fuzzy rewards
    φ(s) = Σ w_i * r_fuzzy_i(s) where Σ w_i = 1
    Ensures output is in [0,1]
    """
    # Weights for each fuzzy reward component (must sum to 1)
    weights = {
        'em': 0.3,      # Exact match has higher weight
        'bm25': 0.15,   # Text similarity
        'f1': 0.15,     # Overall performance
        'recall': 0.15, # Coverage
        'precision': 0.15, # Accuracy
        'sbert': 0.1    # Semantic similarity
    }
    
    # Ensure weights sum to 1
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        print(f"Warning: Weights sum to {weight_sum}, normalizing...")
        weights = {k: v/weight_sum for k, v in weights.items()}
    
    # Compute potential using fuzzy rewards
    potential = 0.0
    for reward_type, value in reward_scores.items():
        fuzzy_reward = compute_fuzzy_reward(reward_type, value)
        potential += weights[reward_type] * fuzzy_reward
    
    # Potential is already in [0,1] since weights sum to 1 and fuzzy_rewards in [0,1]
    return potential

def get_state_signature(prediction, answer=None):
    """
    Generate a unique state signature for tracking s and s'
    Includes both the prediction and extracted answer to better represent state
    """
    if isinstance(prediction, list):
        prediction = prediction[-1]  # Use last prediction for state
    state_str = f"{prediction}|{answer if answer else ''}"
    return hash(state_str)

def compute_score(prediction: list, ground_truth: list, extra_info=None):
    # Extract answer
    assert len(prediction) > 0, "[Error Occurred] Model Responses are empty!"
    print("[Debug] Raw prediction:", prediction[-1])
    answer = extract_solution(prediction=prediction[-1])
    print("[Debug] Extracted answer:", answer)
    
    # Initialize scores
    reward_scores = {
        'em': 0,
        'bm25': 0,
        'f1': 0,
        'recall': 0,
        'precision': 0,
        'sbert': 0
    }
    
    # Compute raw rewards (R_kwd in the paper)
    if answer is not None:
        # Update counters
        for key in reward_call_counts:
            reward_call_counts[key] += 1
            
        # Original EM/SubEM check
        reward_mode = extra_info.get('reward_mode', 'EM') if extra_info else 'EM'
        if reward_mode == "EM" and em_check(answer, ground_truth):
            reward_scores['em'] = 1
        elif reward_mode == 'SubEM' and subem_check(answer, ground_truth):
            reward_scores['em'] = 1
            
        # Compute additional rewards
        for gt in ground_truth:
            print("[Debug] Processing ground truth:", gt)
            reward_scores['bm25'] = max(reward_scores['bm25'], compute_bm25_score(answer, gt))
            reward_scores['f1'] = max(reward_scores['f1'], compute_f1_score(answer, gt))
            reward_scores['recall'] = max(reward_scores['recall'], compute_recall_score(answer, gt))
            reward_scores['precision'] = max(reward_scores['precision'], compute_precision_score(answer, gt))
            reward_scores['sbert'] = max(reward_scores['sbert'], compute_sbert_cosine_score(answer, gt))
            print("[Debug] Current reward scores:", reward_scores)
    else:
        print("[Warning] No answer extracted from prediction")
    
    # Update running statistics and normalize rewards
    for reward_type, value in reward_scores.items():
        update_running_stats(reward_type, value)
        reward_scores[reward_type] = normalize_reward(reward_type, value)
    
    # Generate state signatures for s and s'
    current_state = get_state_signature(prediction, answer)
    
    # Compute potential-based shaping
    # φ(s) = Σ w_i * r_fuzzy_i(s)
    current_potential = compute_potential(current_state, reward_scores)
    
    # Get previous state's potential, defaulting to 0 if not found
    prev_potential = previous_states.get(current_state, 0)
    
    # Update state tracking for future use
    previous_states[current_state] = current_potential
    
    # Compute shaped reward using the formula:
    # R_shaped = R_kwd + γ * φ(s') - φ(s)
    # Ensure non-negative reward through proper scaling
    gamma = 0.99  # discount factor ∈ [0,1]
    base_reward = sum(reward_scores.values()) / len(reward_scores)  # R_kwd
    
    # Scale the potential difference to prevent negative rewards
    potential_diff = gamma * current_potential - prev_potential
    # Map potential_diff to [0,1] using sigmoid
    scaled_potential_diff = 2 / (1 + math.exp(-potential_diff)) - 1  # Maps to [-1,1]
    
    # Combine base reward with scaled potential difference
    alpha = 0.7  # Weight for base reward vs potential difference
    shaped_reward = alpha * base_reward + (1 - alpha) * (1 + scaled_potential_diff) / 2
    
    # Ensure final reward is in [0,1]
    shaped_reward = min(max(shaped_reward, 0), 1)

    # Format check
    format_score, search_count = format_reward(prediction)
    
    # Search penalty
    search_penalty = extra_info.get('search_penalty', 0.1) if extra_info else 0.1
    format_penalty = extra_info.get('format_penalty', 0.1) if extra_info else 0.1
    
    # Apply search penalty
    if search_count > 0 and shaped_reward > 0.99:
        use_search_count_penalty = extra_info.get('use_search_count_penalty', False) if extra_info else False
        if use_search_count_penalty:
            for _ in range(search_count):
                shaped_reward *= 1 - search_penalty
        else:
            shaped_reward *= 1 - search_penalty
    
    # Log to wandb
    if wandb.run is not None:
        step = max(reward_call_counts.values())
        wandb.log({
            'reward/bm25': reward_scores['bm25'],
            'reward/f1': reward_scores['f1'],
            'reward/recall': reward_scores['recall'],
            'reward/precision': reward_scores['precision'],
            'reward/sbert': reward_scores['sbert'],
            'reward/em': reward_scores['em'],
            'reward/shaped': shaped_reward,
            'reward/format': format_score,
            'reward/search_count': search_count,
            'reward/potential': current_potential,
            'reward/fuzzy_potential': current_potential,  # Log fuzzy potential
            'reward/base': base_reward,  # Log original reward
            'reward/potential_diff': potential_diff,
            'reward/scaled_potential_diff': scaled_potential_diff  # Log the scaled difference
        }, step=step)
    
    # Return final shaped reward with format consideration
    final_reward = (1 - format_penalty) * shaped_reward + format_penalty * format_score
    return min(max(final_reward, 0), 1)  # Final safety clamp to [0,1]
