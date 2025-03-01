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

def upsample_attention_map(tensor, target_dim=48):
    """
    upsample the attention map
    Args:
        tensor (Tensor): (B, Q, K)
        target_feature_dim (int): target feature dimension
    """
    B, Q, K = tensor.shape
    W = int(K**0.5)
    tensor = tensor.view(B, Q, W, W)
    tensor = F.interpolate(tensor.float().detach(), size = (target_dim, target_dim), mode='bicubic')
    tensor = tensor.view(B, Q, target_dim*target_dim)
    return tensor


def downsample_patch(tensor, target_patch):
    original_shape = tensor.shape[:-2]
    tensor = tensor.view(-1, 1, *tensor.shape[-2:]) # (..., W, H) -> (-1, 1, W, H)
    
    tensor = F.interpolate(tensor, size=(target_patch, target_patch), mode='bilinear', align_corners=False)
    return tensor.view(*original_shape, target_patch, target_patch)

    
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


def rollout(prev_rollout, attention_map):
    """
    Args:
        prev_rollout (Tensor): 이전 rollout된 어텐션 맵 (Batch, Q, K)
        attention_map (Tensor): 현재 어텐션 맵 (Batch, Q, K)
    """
    B,Q, K = attention_map.shape
    identity = torch.eye(K, device=attention_map.device, dtype=attention_map.dtype).expand_as(attention_map)

    if prev_rollout is None:
        print("this should be first block")
        return attention_map
    else:
        return torch.matmul(attention_map+identity, prev_rollout)

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


def rollout_cross_attention_map(attention_weight, device, head_fusion="mean", downsample=24, prev_rollout=None ):
    """
    rolling out the image cross attention map, with pure cross-attention weight.
    Batch개의 Attention map을 반환
    """
    B, H, Q, K = attention_weight.shape
    prev_rollout = prev_rollout.to(device) if prev_rollout is not None else None
    attention_weight = fuse_heads(attention_weight, head_fusion) # (B, Q, K) = (B, W*W, W*W)
    attention_weight = downsample_cross_attention(attention_weight, downsample) # (B, 576, 576)
    return sum_up_attention_map(prev_rollout, attention_weight)
    #return rollout(prev_rollout, attention_weight)

def rollout_self_attention_map(attention_weight, device, head_fusion="mean", downsample=24, prev_rollout=None):
    """
    rolling out the row wise self attention map
    """
    B, H, Q, K = attention_weight.shape # B H (ih iw) (nv iw)
    prev_rollout = prev_rollout.to(device) if prev_rollout is not None else None
    attention_weight = fuse_heads(attention_weight, head_fusion) # (B, Q, K) = (B, h*w, B*w)
    attention_weight = downsample_self_attention(attention_weight, downsample) # (B, 24*24, B*24)
    return sum_up_attention_map(prev_rollout, attention_weight)


def get_heatmap_from_key_patch(attention_weight, selected_patch=0):
    """
    Args:
        attention_weight (Tensor): (B, Q, K)
        selected_patch (int): patch index of key(reference) image
    output:
        heatmap (Tensor): (B, Q) B of heatmaps for selected patch from reference image 
    """
    B, Q, K = attention_weight.shape
    heatmaps = attention_weight[:, :, selected_patch] # (B, Q) NEED TO CHECK. Whether to visualize Q or K
    heatmaps = rearrange(heatmaps, 'b (w h) -> b w h', w=int(Q**0.5), h=int(Q**0.5)) #(B, Q) -> (B, W, H)
    return heatmaps # (B, W, H)

def get_heatpmap_from_query_patch(attention_weight, selected_view=0, selected_patch=0):
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

def get_heatmap_from_query_column(attention_weight, selected_view, selected_column):
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

def visualize_heatmap(mask, dirname="mask", ref_image_path=None, step=0,  save=False, need_display=True, minmaxscale=True):
    """
    Args:
        mask (Tensor): (B, W, H) 형태의 attention mask (값 범위 [0,1])
    """
    #mask = mask.detach().cpu().numpy()  # GPU → CPU 변환
    if mask.ndim == 2:
        mask = mask[np.newaxis, :, :]  # (W, H) → (1, W, H)
    B, W, H = mask.shape  # 배치 크기 유지

    # NaN 및 Inf 값 처리
    mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)

    if(minmaxscale):
        mask_min = np.min(mask, axis=(1, 2), keepdims=True)
        mask_max = np.max(mask, axis=(1, 2), keepdims=True)
        mask = np.where(mask_max == mask_min, mask, (mask - mask_min) / (mask_max - mask_min))

    # uint8 변환 후 컬러맵 적용 (BGR로 생성됨)
    heatmap = [cv2.applyColorMap(np.uint8(m * 255), cv2.COLORMAP_JET) for m in mask]

    # 해상도 증가 (8배 확대) 및 uint8 변환 유지
    heatmap = [cv2.resize(hm, (H * 8, W * 8), interpolation=cv2.INTER_CUBIC) for hm in heatmap]

    # 배치 차원 유지 → 가로로 병합 (W*8, H*B*8)
    heatmap = np.hstack(heatmap)  # (W*8, H*B*8, 3)

    if ref_image_path is not None:
        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.resize(ref_image, (H * 8, W * 8), interpolation=cv2.INTER_CUBIC)
        heatmap = np.float32(heatmap)*0.8 + np.float32(ref_image)*0.2
        heatmap = np.uint8(heatmap / np.max(heatmap)*255)

    # PNG로 저장하는 이미지와 display에서 보여주는 이미지가 동일하게 설정
    if save:
        save_dir = f"attn_maps/{dirname}/"
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(f"{save_dir}{step:02d}.png", heatmap)

    if need_display:
        clear_output(wait=True)
        img_pil = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))  # OpenCV BGR → RGB 변환
        display(img_pil)  # 바로 표시
    