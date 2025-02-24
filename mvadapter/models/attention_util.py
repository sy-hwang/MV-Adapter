import torch
import torch.nn.functional as F
import numpy as np

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Scaled Dot-Product Attention을 직접 구현하여 어텐션 가중치를 반환하는 함수
    
    Args:
        q (Tensor): Query 행렬 (Batch, Heads, Query Length, Dim)
        k (Tensor): Key 행렬 (Batch, Heads, Key Length, Dim)
        v (Tensor): Value 행렬 (Batch, Heads, Key Length, Dim)
        mask (Tensor, optional): 어텐션 마스크 (Broadcast 가능 형태)
        
    Returns:
        output (Tensor): 어텐션이 적용된 값 (Batch, Heads, Query Length, Dim)
        attention_weights (Tensor): Softmax(QK^T) 어텐션 가중치 (Batch, Heads, Query Length, Key Length)
    """
    # Query와 Key의 차원 크기(Dim)를 가져오기
    d_k = q.size(-1)

    # (Q @ K^T) / sqrt(d_k) 연산 수행
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=q.dtype))

    # 마스크가 있을 경우 적용 (예: 패딩 마스크 또는 캐주얼 마스크)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))

    # Softmax를 적용하여 확률값으로 변환 (어텐션 가중치 계산)
    attention_weights = F.softmax(attn_logits, dim=-1)

    # 어텐션 가중치와 Value 행렬을 곱하여 최종 출력 계산
    output = torch.matmul(attention_weights, v)

    return output, attention_weights

def compute_attention_rollout(attn_maps):
    """
    Attention Rollout을 계산하는 함수.

    Args:
        attn_maps (List[np.ndarray] 또는 List[torch.Tensor]): 
            - shape: (num_layers, 48, 48) 형태의 attention maps 리스트
            - self.full_attention_maps을 입력으로 받음

    Returns:
        torch.Tensor: Attention Rollout 결과 (48x48)
    """

    # 리스트가 비어있는 경우 예외 처리
    if len(attn_maps) == 0:
        raise ValueError("Error: Attention maps list is empty!")

    # NumPy 배열이 들어온 경우 PyTorch Tensor로 변환
    if isinstance(attn_maps[0], np.ndarray):
        attn_maps = [torch.tensor(a, dtype=torch.float32) for a in attn_maps]

    # (num_layers, 48, 48) 형태로 변환
    attn_maps = torch.stack(attn_maps)  # (L, 48, 48)

    # Identity Matrix 추가 (Skip Connection 효과)
    identity = torch.eye(48, dtype=attn_maps.dtype, device=attn_maps.device)  # (48, 48)

    # Attention 정규화 (각 행의 합이 1이 되도록)
    attn_maps = attn_maps + identity
    attn_maps /= attn_maps.sum(dim=-1, keepdim=True)  # Row-wise normalize

    # Attention Rollout: 모든 레이어를 행렬 곱으로 누적
    attn_rollout = attn_maps[0]  # 첫 번째 레이어 초기값
    for i in range(1, attn_maps.shape[0]):
        attn_rollout = torch.matmul(attn_rollout, attn_maps[i])  # (48, 48)

    return attn_rollout
