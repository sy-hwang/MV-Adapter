import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def visualize_patch_attention(cross_attn_weights, filename, head_fusion="mean"):
    """
    cross_attn_weights: 여러 레이어에서 수집한 attention weight 리스트.
                         각 원소의 shape는 (B, Heads, Query Length)
    filename: 저장할 파일 이름 (확장자는 자동으로 ".png"가 붙음)
    head_fusion: 융합 방식 ("mean" 또는 "max")
    
    주의: 각 레이어마다 query length가 다를 수 있으므로,
         모든 레이어 중 최대 query length를 기준으로 리사이징한 후 rollout 적용합니다.
         (불필요한 변수 할당 없이 in-place 연산 위주로 구현)
    """
    import numpy as np
    from PIL import Image
    import math
    import matplotlib.cm as cm

    # 배치 크기와 최대 query length 결정 (최소한의 변수만 사용)
    first = cross_attn_weights[0]
    if hasattr(first, 'detach'):
        first = first.detach().cpu().numpy()
    B = first.shape[0]
    max_query_length = 0
    for attn in cross_attn_weights:
        if hasattr(attn, 'detach'):
            attn = attn.detach().cpu().numpy()
        cur_len = attn.shape[-1]
        if cur_len > max_query_length:
            max_query_length = cur_len
    max_side = int(math.sqrt(max_query_length))
    if max_side * max_side != max_query_length:
        raise ValueError("최대 query length (L={})가 정사각형 형태가 아닙니다.".format(max_query_length))
    
    # rollout 초기화 (배치별, 최대 query length)
    rollout = np.ones((B, max_query_length), dtype=np.float32)

    # 각 레이어 처리: head fusion 후 바로 rollout에 반영
    for attn in cross_attn_weights:
        if hasattr(attn, 'detach'):
            attn = attn.detach().cpu().numpy()
        # head fusion (B, Query Length)
        if head_fusion == "mean":
            fused = np.mean(attn, axis=1)
        elif head_fusion == "max":
            fused = np.max(attn, axis=1)
        elif head_fusion == "min":
            fused = np.min(attn, axis=1)
        else:
            raise ValueError("Unsupported head fusion method: {}".format(head_fusion))
        current_query_length = fused.shape[1]
        current_side = int(math.sqrt(current_query_length))
        if current_side * current_side != current_query_length:
            raise ValueError("어떤 레이어의 query length (L={})가 정사각형 형태가 아닙니다.".format(current_query_length))
        
        # 크기가 다르면 배치별로 개별 resize 후 rollout에 곱셈 반영
        if current_query_length != max_query_length:
            for i in range(B):
                cur_map = fused[i].reshape(current_side, current_side)
                im = Image.fromarray(cur_map.astype(np.float32), mode='F')
                resized_im = im.resize((max_side, max_side), resample=Image.BILINEAR)
                rollout[i] *= np.array(resized_im).flatten()
        else:
            rollout *= fused  # 크기가 같으면 바로 곱셈

    # 최종 rollout 결과를 배치별로 정규화하여 하나의 이미지에 좌우로 붙임
    colormap = cm.get_cmap('jet')
    final_img = Image.new("RGB", (B * max_side, max_side))
    for i in range(B):
        attn_map = rollout[i].reshape(max_side, max_side)
        attn_map = attn_map/np.max(attn_map)
        colored = (colormap(attn_map)[..., :3] * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(colored, mode="RGB")
        final_img.paste(heatmap_img, (i * max_side, 0))
    
    final_img.save(filename + "rgb.png")
