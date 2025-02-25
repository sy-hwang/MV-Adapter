import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from IPython.display import clear_output, display
from PIL import Image

def downsample_patch(tensor, target_patch):
    B, W, H = tensor.shape
    tensor = tensor.unsqueeze(1) # (B, 1, W, H)
    tensor = F.interpolate(tensor, size=(target_patch, target_patch), mode='bilinear', align_corners=False)
    return tensor.squeeze(1) #(B, target, target)

    
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
        prev_rollout (Tensor): 이전에 rollout된 어텐션 맵 (Batch, W, H)
        attention_map (Tensor): 현재 어텐션 맵 (Batch, W, H)
    """
    B, W, H = attention_map.shape
    if prev_rollout is None:
        return attention_map
    else:
        identity = torch.eye(W, device=attention_map.device, dtype=attention_map.dtype).expand(B, W, H)
        return torch.bmm(attention_map+identity, prev_rollout)



import numpy as np
import cv2
from PIL import Image
from IPython.display import display, clear_output

import numpy as np
import cv2
from PIL import Image
from IPython.display import display, clear_output

def show_mask_on_image(mask, img_path, filename="mask", save=True, need_display=True):
    """
    Args:
        mask (Tensor): (B, W, H) 형태의 attention mask (값 범위 [0,1])
        img_path (str): 원본 이미지 경로 (현재 사용되지 않음)
        filename (str): 저장할 파일명
    """
    mask = mask.detach().cpu().numpy()  # GPU → CPU 변환
    B, W, H = mask.shape  # 배치 크기 유지

    # NaN 및 Inf 값 처리
    mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)

    # 값이 모두 0이면 대비 조정 (최소 0, 최대 1 설정)
    mask_min = np.min(mask, axis=(1, 2), keepdims=True)
    mask_max = np.max(mask, axis=(1, 2), keepdims=True)
    mask = np.where(mask_max == mask_min, mask + 1e-6, (mask - mask_min) / (mask_max - mask_min))  # Min-Max Scaling

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

