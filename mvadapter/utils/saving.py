import math
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

import os
import IPython.display as display


def tensor_to_image(
    data: Union[Image.Image, torch.Tensor, np.ndarray],
    batched: bool = False,
    format: str = "HWC",
) -> Union[Image.Image, List[Image.Image]]:
    if isinstance(data, Image.Image):
        return data
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if data.dtype == np.float32 or data.dtype == np.float16:
        data = (data * 255).astype(np.uint8)
    elif data.dtype == np.bool_:
        data = data.astype(np.uint8) * 255
    assert data.dtype == np.uint8
    if format == "CHW":
        if batched and data.ndim == 4:
            data = data.transpose((0, 2, 3, 1))
        elif not batched and data.ndim == 3:
            data = data.transpose((1, 2, 0))

    if batched:
        return [Image.fromarray(d) for d in data]
    return Image.fromarray(data)


def largest_factor_near_sqrt(n: int) -> int:
    """
    Finds the largest factor of n that is closest to the square root of n.

    Args:
        n (int): The integer for which to find the largest factor near its square root.

    Returns:
        int: The largest factor of n that is closest to the square root of n.
    """
    sqrt_n = int(math.sqrt(n))  # Get the integer part of the square root

    # First, check if the square root itself is a factor
    if sqrt_n * sqrt_n == n:
        return sqrt_n

    # Otherwise, find the largest factor by iterating from sqrt_n downwards
    for i in range(sqrt_n, 0, -1):
        if n % i == 0:
            return i

    # If n is 1, return 1
    return 1


def make_image_grid(
    images: List[Image.Image],
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    resize: Optional[int] = None,
) -> Image.Image:
    """
    Prepares a single grid of images. Useful for visualization purposes.
    """
    if rows is None and cols is not None:
        assert len(images) % cols == 0
        rows = len(images) // cols
    elif cols is None and rows is not None:
        assert len(images) % rows == 0
        cols = len(images) // rows
    elif rows is None and cols is None:
        rows = largest_factor_near_sqrt(len(images))
        cols = len(images) // rows

    assert len(images) == rows * cols

    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

from PIL import Image, ImageDraw
import numpy as np

from PIL import Image, ImageDraw

def draw_patches(image, num_patches=(24, 24), highlight_index=None, highlight_column=None,
                                line_color=(255, 255, 255), line_width=2,
                                highlight_color=(255, 0, 0), highlight_width=4):
    """
    입력 이미지를 patch 개수에 맞춰 나누고, 특정 패치를 강조하는 함수.

    Args:
        image (PIL.Image): 입력 이미지
        num_patches (tuple): 패치 개수 (가로 개수, 세로 개수)
        highlight_index (int): 강조할 패치의 리니어 인덱스 (0-based index)
        highlight_column (int): 강조할 패치의 열 인덱스 (0-based index)
        line_color (tuple): 경계선 색상 (RGB, 기본: 흰색)
        line_width (int): 경계선 두께 (기본: 2px)
        highlight_color (tuple): 강조 패치 테두리 색상 (RGB, 기본: 빨간색)
        highlight_width (int): 강조 패치 테두리 두께 (기본: 4px)

    Returns:
        PIL.Image: 패치 경계선을 포함한 새로운 이미지
    """
    img_with_patches = image.copy()
    draw = ImageDraw.Draw(img_with_patches)

    img_width, img_height = image.size
    num_patches_w, num_patches_h = num_patches

    # 패치 크기 계산
    patch_w = img_width // num_patches_w
    patch_h = img_height // num_patches_h

    # 세로줄 그리기
    for x in range(patch_w, img_width, patch_w):
        draw.line([(x, 0), (x, img_height)], fill=line_color, width=line_width)

    # 가로줄 그리기
    for y in range(patch_h, img_height, patch_h):
        draw.line([(0, y), (img_width, y)], fill=line_color, width=line_width)

    # 강조할 패치 찾기
    if highlight_index is not None:
        total_patches = num_patches_w * num_patches_h
        if 0 <= highlight_index < total_patches:
            # 리니어 인덱스를 2D 인덱스로 변환
            x_idx = highlight_index % num_patches_w  # 가로 인덱스
            y_idx = highlight_index // num_patches_w  # 세로 인덱스

            x1, y1 = x_idx * patch_w, y_idx * patch_h
            x2, y2 = x1 + patch_w, y1 + patch_h

            # 강조된 패치 테두리 (빨간색 + 굵은 선)
            draw.rectangle([x1, y1, x2, y2], outline=highlight_color, width=highlight_width)
    
    if highlight_column is not None:
        if 0 <= highlight_column < num_patches_w:
            x_idx = highlight_column
            for y_idx in range(num_patches_h):
                x1, y1 = x_idx * patch_w, y_idx * patch_h
                x2, y2 = x1 + patch_w, y1 + patch_h
                draw.rectangle([x1, y1, x2, y2], outline=highlight_color, width=highlight_width)

    return img_with_patches

from PIL import Image
import os
import IPython.display as display

def sorted_numerically(file_list):
    """ 숫자로 정렬하기 위한 키 함수 """
    return sorted(file_list, key=lambda x: int(os.path.splitext(x)[0]))

def resize_keep_aspect(image, target_height):
    """
    원본 비율을 유지하면서 세로 길이만 설정하고 가로 길이는 자동 조정
    """
    width, height = image.size
    aspect_ratio = width / height
    new_width = int(target_height * aspect_ratio)
    return image.resize((new_width, target_height), Image.Resampling.LANCZOS)

def png_to_gif(input_folder, output_gif, duration=100, loop=0):
    """
    PNG 시퀀스를 GIF로 변환하고 Jupyter Notebook에서 바로 표시하는 함수
    
    :param input_folder: PNG 이미지가 있는 폴더 경로
    :param output_gif: 생성할 GIF 파일 경로
    :param duration: 각 프레임의 지속 시간 (ms 단위, 기본값: 100ms)
    :param loop: GIF 루프 횟수 (0이면 무한 반복)
    """
    # PNG 파일 가져오기 및 숫자 순서대로 정렬
    images = [f for f in os.listdir(input_folder) if f.endswith(".png")]
    images = sorted_numerically(images)
    images = [os.path.join(input_folder, f) for f in images]

    # 이미지 로드
    frames = [resize_keep_aspect(Image.open(img).convert("P", palette=Image.ADAPTIVE), 192) for img in images]
    
    # GIF 저장
    if frames:
        black_frame = Image.new("P", frames[0].size, 0)
        frames.insert(0, black_frame)
        frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=duration, loop=loop, optimize=True)
        print(f"gif 저장 완료: {output_gif}, filesize:{os.path.getsize(output_gif)/1024:.2f}KB")

        # Jupyter Notebook에서 GIF 표시
        display.display(display.Image(output_gif))
    else:
        print("PNG 파일을 찾을 수 없습니다.")

def mask_to_gif(input_folder, output_gif, ref_img, duration=100, loop=0):
    """
    PNG 시퀀스를 GIF로 변환하고 Jupyter Notebook에서 바로 표시하는 함수
    
    :param input_folder: PNG 이미지가 있는 폴더 경로
    :param output_gif: 생성할 GIF 파일 경로
    :param duration: 각 프레임의 지속 시간 (ms 단위, 기본값: 100ms)
    :param loop: GIF 루프 횟수 (0이면 무한 반복)
    """
    # PNG 파일 가져오기 및 숫자 순서대로 정렬
    masks = [f for f in os.listdir(input_folder) if f.endswith(".png")]
    masks = sorted_numerically(masks)
    masks = [os.path.join(input_folder, f) for f in masks]

    ref_img = resize_keep_aspect(ref_img.convert("RGB"), 96)
    ref_img_np = np.array(ref_img, dtype=np.float32)

    frames = []
    for mask in masks:
        mask_img = Image.open(mask).convert("RGB")
        mask_img = resize_keep_aspect(mask_img, 96)
        mask_np = np.array(mask_img, dtype=np.float32)
        blended_np = mask_np*0.8 + ref_img_np*0.2
        blended_np = np.uint8(blended_np/np.max(blended_np)*255)
        blended_img = Image.fromarray(blended_np).convert("P", palette=Image.ADAPTIVE)
        frames.append(blended_img)
    
    # GIF 저장
    if frames:
        black_frame = Image.new("P", frames[0].size, 0)
        frames.insert(0, black_frame)
        frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=duration, loop=loop, optimize=True)
        print(f"gif 저장 완료: {output_gif}, filesize:{os.path.getsize(output_gif)/1024:.2f}KB")

        # Jupyter Notebook에서 GIF 표시
        display.display(display.Image(output_gif))
    else:
        print("PNG 파일을 찾을 수 없습니다.")