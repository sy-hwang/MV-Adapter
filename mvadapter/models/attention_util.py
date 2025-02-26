import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from IPython.display import clear_output, display
from PIL import Image
from einops import rearrange

def downsample_attention(tensor, target_feature_dim=24):
    """
    downsample the attention map
    Args:
        tensor (Tensor): (B, H, Q, K)
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



def downsample_patch(tensor, target_patch):
    original_shape = tensor.shape[:-2]
    tensor = tensor.view(-1, 1, *tensor.shape[-2:]) # (..., W, H) -> (-1, 1, W, H)
    
    tensor = F.interpolate(tensor, size=(target_patch, target_patch), mode='bilinear', align_corners=False)
    return tensor.view(*original_shape, target_patch, target_patch)

    
def get_attention_weight(q, k, v, dropout_p=0.0, is_causal=False):
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
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=q.dtype))

    # 마스크가 있을 경우 적용 (예: 패딩 마스크 또는 캐주얼 마스크)
    if is_causal:
        seq_len = q.shape[-2]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(q.device)  # Upper triangular matrix
        attn_logits = attn_logits.masked_fill(causal_mask == 1, float('-inf'))  # 미래 정보를 가려줌
    
    # Softmax를 적용하여 확률값으로 변환 (어텐션 가중치 계산)
    attention_weights = F.softmax(attn_logits, dim=-1)

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
        return attention_map
    else:
        return torch.matmul(attention_map+identity, prev_rollout)

def rollout_cross_attention_map(attention_weight, head_fusion="mean", downsample=24, prev_rollout=None, device='cuda'):
    """
    rolling out the image cross attention map, with pure cross-attention weight.
    Batch개의 Attention map을 반환
    """
    B, H, Q, K = attention_weight.shape
    prev_rollout = prev_rollout.to(device) if prev_rollout is not None else None
    attention_weight = fuse_heads(attention_weight, head_fusion) # (B, Q, K) = (B, W*W, W*W)
    attention_weight = downsample_attention(attention_weight, 24) # (B, 576, 576)
    return rollout(prev_rollout, attention_weight)

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

def visualize_heatmap(mask, filename="mask", save=False, need_display=True):
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

    # 값이 모두 0이면 대비 조정 (최소 0, 최대 1 설정)
    mask_min = np.min(mask, axis=(1, 2), keepdims=True)
    mask_max = np.max(mask, axis=(1, 2), keepdims=True)
    mask = np.where(mask_max == mask_min, mask, (mask - mask_min) / (mask_max - mask_min))  # Min-Max Scaling

    # uint8 변환 후 컬러맵 적용 (BGR로 생성됨)
    heatmaps = [cv2.applyColorMap(np.uint8(m * 255), cv2.COLORMAP_JET) for m in mask]

    # 해상도 증가 (8배 확대) 및 uint8 변환 유지
    heatmaps_resized = [cv2.resize(hm, (H * 8, W * 8), interpolation=cv2.INTER_CUBIC) for hm in heatmaps]

    # 배치 차원 유지 → 가로로 병합 (W*8, H*B*8)
    combined_heatmap = np.hstack(heatmaps_resized)  # (W*8, H*B*8, 3)

    # PNG로 저장하는 이미지와 display에서 보여주는 이미지가 동일하게 설정
    if save:
        cv2.imwrite(f"attn_maps/heat-{filename}.png", combined_heatmap)
        print(f"✅ Heatmap saved to attn_maps/heat-{filename}.png")

    if need_display:
        clear_output(wait=True)
        img_pil = Image.fromarray(cv2.cvtColor(combined_heatmap, cv2.COLOR_BGR2RGB))  # OpenCV BGR → RGB 변환
        display(img_pil)  # 바로 표시
    

def rollout_cross_attention_map_1(attention_weight, head_fusion="mean", selected_patch=0, downsample=24, prev_rollout=None, device='cuda'):
    """
    rolling out the image cross attention map, with pure cross-attention weight.
    reference image(key)를 기준으로 query image(mv images)의 attention map을 시각화하는 함수
    Args:
        attention_weight (Tensor): (B, H, Q, K)
        head_fusion (str): 헤드 통합 방법 (mean, max, min)
        selected_patch (int): 선택한 패치 번호 (downsample*downsample 보다 작아야 함)
        downsample (int): 다운샘플링 크기
    """
    B, H, Q, K = attention_weight.shape
    attention_weight = fuse_heads(attention_weight, head_fusion) # (B, Q, K)
    attention_weight = attention_weight[:, :, selected_patch] # (B, Q) NEED TO CHECK. Whether to visualize Q or K
    W = int(Q**0.5)
    attention_weight = rearrange(attention_weight, 'b (w h) -> b w h', w=W, h=W) #(B, Q) -> (B, W, H)
    attention_weight = downsample_patch(attention_weight, downsample).to(device) # (B, downsample, downsample)
    return rollout(prev_rollout, attention_weight)

def rollout_cross_attention_map_2(attention_weight, head_fusion="mean", selected_view=0,selected_patch=0, downsample=24, prev_rollout=None, device='cuda'):
    """
    rolling out the image cross attention map, with pure cross-attention weight.
    mv images들 중 하나(query)를 기준으로 reference image(key)의 attention map을 시각화하는 함수
    Args:
        attention_weight (Tensor): (B, H, Q, K)
        head_fusion (str): 헤드 통합 방법 (mean, max, min)
        selected_view (int): 선택한 mv image
        selected_patch (int): 선택한 패치 번호 (downsample*downsample 보다 작아야 함)
        downsample (int): 다운샘플링 크기
    """
    B, H, Q, K = attention_weight.shape
    attention_weight = fuse_heads(attention_weight, head_fusion) # (B, Q, K)
    attention_weight = attention_weight[selected_view, selected_patch, :] # (K)
    W = int(K**0.5)
    attention_weight = rearrange(attention_weight, '(w h) -> w h', w=W, h=W) #(K) -> (W, H)
    attention_weight = downsample_patch(attention_weight, downsample).to(device) # (downsample, downsample)
    return rollout(prev_rollout, attention_weight)




def show_mask_on_image(mask, img_path, filename="mask", save=True, need_display=True):
    """
    Args:
        mask (Tensor): (B, W, H), 또는 (W, H) 형태의 attention mask (값 범위 [0,1])
        img_path (str): 원본 이미지 경로 (현재 사용되지 않음)
        filename (str): 저장할 파일명
    """
    mask = mask.detach().cpu().numpy()  # GPU → CPU 변환
    if mask.ndim == 2:
        mask = mask[np.newaxis, :, :]  # (W, H) → (1, W, H)
    B, W, H = mask.shape  # 배치 크기 유지

    # NaN 및 Inf 값 처리
    mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)

    # 값이 모두 0이면 대비 조정 (최소 0, 최대 1 설정)
    mask_min = np.min(mask, axis=(1, 2), keepdims=True)
    mask_max = np.max(mask, axis=(1, 2), keepdims=True)
    mask = np.where(mask_max == mask_min, mask, (mask - mask_min) / (mask_max - mask_min))  # Min-Max Scaling

    # uint8 변환 후 컬러맵 적용 (BGR로 생성됨)
    heatmaps = [cv2.applyColorMap(np.uint8(m * 255), cv2.COLORMAP_JET) for m in mask]

    # 해상도 증가 (8배 확대) 및 uint8 변환 유지
    heatmaps_resized = [cv2.resize(hm, (H * 8, W * 8), interpolation=cv2.INTER_CUBIC) for hm in heatmaps]

    # 배치 차원 유지 → 가로로 병합 (W*8, H*B*8)
    combined_heatmap = np.hstack(heatmaps_resized)  # (W*8, H*B*8, 3)

    # ✅ PNG로 저장하는 이미지와 display에서 보여주는 이미지가 동일하게 설정
    if save:
        cv2.imwrite(f"attn_maps/heat-{filename}.png", combined_heatmap)
        print(f"✅ Heatmap saved to attn_maps/heat-{filename}.png")

    if need_display:
        clear_output(wait=True)
        img_pil = Image.fromarray(cv2.cvtColor(combined_heatmap, cv2.COLOR_BGR2RGB))  # OpenCV BGR → RGB 변환
        display(img_pil)  # 바로 표시

