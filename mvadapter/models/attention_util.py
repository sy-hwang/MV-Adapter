import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from IPython.display import clear_output, display
from PIL import Image
from einops import rearrange
import os

def downsample_cross_attention(tensor, target_feature_dim=24):
    """
    downsample the attention map
    Args:
        tensor (Tensor): (B, Q, K)
        target_feature_dim (int): target feature dimension
    """
    B, Q, K = tensor.shape
    W = H = int(Q**0.5)
    if W == target_feature_dim:
        return tensor
    
    scale = W//target_feature_dim

    #key 차원 다운샘플링
    tensor = tensor.view(B, Q, W, H)
    tensor = tensor.view(B, Q, target_feature_dim, scale, target_feature_dim, scale)
    tensor = tensor.mean(dim=(-1, -3)) # (B, Q, target_feature_dim, target_feature_dim)
    tensor = tensor.view(B, Q, target_feature_dim*target_feature_dim) # (B, Q, target_feature_dim*target_feature_dim)
    K = target_feature_dim*target_feature_dim

    #Query 차원 다운샘플링
    tensor = tensor.view(B, W, H, K)
    tensor = tensor.view(B, target_feature_dim, scale, target_feature_dim, scale, K)
    tensor = tensor.mean(dim=(-2, -4)) # (B, target_feature_dim, target_feature_dim, K)
    tensor = tensor.view(B, target_feature_dim*target_feature_dim, K) # (B, target_feature_dim*target_feature_dim, K)
    Q = target_feature_dim*target_feature_dim

    return tensor.view(B, Q, K)

def downsample_self_attention(tensor, target_feature_dim=24):
    """
    downsample the attention map
    Args:
        tensor (Tensor): (B, Q, K) = (B, h*w, B*w)
        target_feature_dim (int): target feature dimension
    """
    B, Q, K = tensor.shape
    W = H = int(Q**0.5)
    if W == target_feature_dim:
        return tensor
    
    scale = W//target_feature_dim

    #key 차원 다운샘플링
    tensor = tensor.view(B, Q, B, W)
    tensor = tensor.view(B, Q, B, target_feature_dim, scale)
    tensor = tensor.mean(dim=-1) # (B, Q, B, target_feature_dim)
    tensor = tensor.view(B, Q, B*target_feature_dim) # (B, Q, B*target_feature_dim)
    K = B*target_feature_dim

    #Query 차원 다운샘플링
    tensor = tensor.view(B, H, W, K)
    tensor = tensor.view(B, target_feature_dim, scale, target_feature_dim, scale, K)
    tensor = tensor.mean(dim=(-2, -4)) # (B, target_feature_dim, target_feature_dim, K)
    tensor = tensor.view(B, target_feature_dim*target_feature_dim, K) # (B, target_feature_dim*target_feature_dim, K)
    Q = target_feature_dim*target_feature_dim
    
    return tensor.view(B, Q, K)
    
def get_attention_weight(q, k, v, dropout_p=0.0, is_causal=False, use_softmax=True):
    """
    Scaled Dot-Product Attention을 직접 구현하여 어텐션 가중치를 반환하는 함수
    
    Args:
        query: [batch_size, num_heads, seq_len, d_k] 형태의 쿼리 행렬
        key: [batch_size, num_heads, seq_len, d_k] 형태의 키 행렬
        value: [batch_size, num_heads, seq_len, d_v] 형태의 값 행렬
        dropout_p: 드롭아웃 확률
        is_causal: 트라이앵글 마스킹 적용 여부 (미래 정보 차단)
        
    Returns:
        output (Tensor): 어텐션이 적용된 값 (Batch, Heads, Query Length, Dim)
        attention_weights (Tensor): Softmax(QK^T) 어텐션 가중치 (Batch, Heads, Query Length, Key Length)
    """
    # Query와 Key의 차원 크기(Dim)를 가져오기
    d_k = q.size(-1)

    # (Q @ K^T) / sqrt(d_k) 연산 수행
    attention_weights = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=q.dtype))

    # 마스크가 있을 경우 적용 (예: 패딩 마스크 또는 캐주얼 마스크)
    if is_causal:
        seq_len = q.shape[-2]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(q.device)  # Upper triangular matrix
        attention_weights = attention_weights.masked_fill(causal_mask == 1, float('-inf'))  # 미래 정보를 가려줌
    
    # Softmax를 적용하여 확률값으로 변환 (어텐션 가중치 계산)
    if use_softmax:
        attention_weights = F.softmax(attention_weights, dim=-1)

    if dropout_p > 0.0:
        attention_weights = F.dropout(attention_weights, p=dropout_p)

    return attention_weights

def fuse_heads(attention_map, head_fusion="mean"):
    """
    여러 개의 어텐션 헤드를 하나로 통합하는 함수
    
    Args:
        attention_map (Tensor): 어텐션 맵 (Batch, Heads, ...)
        head_fusion (str): 헤드 통합 방법 (mean, max, min)
    
    Returns:
        output (Tensor): 통합된 어텐션 맵 (Batch, ...)
    """
    if head_fusion == "mean":
        return attention_map.mean(dim=1)
    elif head_fusion == "max":
        return attention_map.max(dim=1).values
    elif head_fusion == "min":
        return attention_map.min(dim=1).values
    else:
        raise ValueError(f"Unsupported head fusion method: {head_fusion}")

def sum_up_attention_map(prev_rollout, attention_map):
    """
    Args:
        prev_rollout (Tensor): 이전 rollout된 어텐션 맵 (Batch, Q, K)
        attention_map (Tensor): 현재 어텐션 맵 (Batch, Q, K)
    """
    if(prev_rollout is None):
        return attention_map
    else:
        return attention_map + prev_rollout


def compute_cross_attention_weight(attention_weight, device, head_fusion="mean", downsample=24, prev_weight=None ):
    """
    rolling out the image cross attention map, with pure cross-attention weight.
    Batch개의 Attention map을 반환
    """
    B, H, Q, K = attention_weight.shape
    prev_weight = prev_weight.to(device) if prev_weight is not None else None
    attention_weight = fuse_heads(attention_weight, head_fusion) # (B, Q, K) = (B, W*W, W*W)
    attention_weight = downsample_cross_attention(attention_weight, downsample) # (B, 576, 576)
    return sum_up_attention_map(prev_weight, attention_weight)

def compute_self_attention_weight(attention_weight, device, head_fusion="mean", downsample=24, prev_weight=None):
    """
    rolling out the row wise self attention map
    """
    B, H, Q, K = attention_weight.shape # B H (ih iw) (nv iw)
    prev_weight = prev_weight.to(device) if prev_weight is not None else None
    attention_weight = fuse_heads(attention_weight, head_fusion) # (B, Q, K) = (B, h*w, B*w)
    attention_weight = downsample_self_attention(attention_weight, downsample) # (B, 24*24, B*24)
    return sum_up_attention_map(prev_weight, attention_weight)

def extract_attention_map_from_query_patch(attention_weight, selected_view=0, selected_patch=0):
    """
    Args:
        attention_weight (Tensor): (B, Q, K)
        selected_view (int): selected mv image
        selected_patch (int): patch index of query(mv) image
    output:
        heatmap (Tensor): (B, K) B of heatmaps for selected patch from query image 
    """
    B, Q, K = attention_weight.shape
    heatmap = attention_weight[selected_view, selected_patch, :] # (K)
    heatmap = rearrange(heatmap, '(w h) -> w h', w=int(K**0.5), h=int(K**0.5)) #(K) -> (W, H)
    return heatmap # (W, H)

def extract_attention_map_from_query_column(attention_weight, selected_view, selected_column):
    """
    Args:
        attention_weight (Tensor): (B, Q, K) = (B, 24*24, B*24)
        selected_view (int): selected mv image
        selected_column (int): column index of query(mv) image
    output:
        heatmap (Tensor): (B, K) B of heatmaps for selected patch from query image 
    """
    B, Q, K = attention_weight.shape
    W = H= int(Q**0.5)
    attention_weight = attention_weight.view(B, H, W, B, W)
    heatmap = attention_weight[selected_view, :, selected_column, :, :]
    heatmap = rearrange(heatmap, 'h b w -> b h w')
    return heatmap