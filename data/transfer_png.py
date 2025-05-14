import os
import h5py
import numpy as np
from glob import glob
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

# 입력/출력 경로
input_root = "/mnt/MW/data/BraTs2020"
output_root = "/mnt/MW/VQGAN_BraTs/data/BraTs2020"
os.makedirs(output_root, exist_ok=True)

contrast_names = ['t1', 't1gd', 't2', 'flair']

# 전체 하위 폴더 포함하여 .h5 파일 찾기
h5_files = sorted(glob(os.path.join(input_root, "**", "volume_*_slice_*.h5"), recursive=True))

# volume_x 기준으로 그룹핑
volume_slices = defaultdict(list)
for path in h5_files:
    fname = os.path.basename(path)
    vol_id = "_".join(fname.split("_")[:2])  # volume_x
    volume_slices[vol_id].append(path)

# 중심 60% 슬라이스 선택 함수
def get_center_slice_paths(slice_paths, ratio=0.6):
    sorted_paths = sorted(slice_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    total = len(sorted_paths)
    center_count = int(total * ratio)
    start = (total - center_count) // 2
    return sorted_paths[start:start+center_count]

# 이미지 리사이즈와 패딩을 추가하는 함수
def resize_with_padding(image, target_size=(256, 256)):
    # 원본 이미지 크기
    width, height = image.size
    target_width, target_height = target_size

    # 새로운 이미지를 target_size로 생성 (배경색은 0, 즉 검은색)
    new_image = Image.new("L", target_size, 0)  # (L은 grayscale, 배경을 검은색으로 설정)

    # 이미지 크기 비율에 맞춰 크기 조정 (240x240 -> 256x256)
    offset = ((target_width - width) // 2, (target_height - height) // 2)

    # 원본 이미지를 새 이미지 중앙에 붙여넣기
    new_image.paste(image, offset)

    return new_image

# 변환 및 저장
for vol_id, paths in tqdm(volume_slices.items(), desc="Processing volumes"):
    center_paths = get_center_slice_paths(paths, ratio=0.6)
    
    for path in center_paths:
        fname = os.path.basename(path).replace(".h5", "")
        with h5py.File(path, 'r') as f:
            image = f['image'][:]  # shape: (H, W, 4)

        for i, contrast in enumerate(contrast_names):
            img = image[:, :, i].astype(np.float32)

            # 정규화
            img -= img.min()
            img /= (img.max() + 1e-5)
            img *= 255.0
            img = img.clip(0, 255).astype(np.uint8)

            # 256x256으로 리사이징 (패딩 추가)
            img_resized = Image.fromarray(img).convert("L")
            img_resized = resize_with_padding(img_resized, target_size=(256, 256))

            # 저장
            out_path = os.path.join(output_root, f"{fname}_{contrast}.png")
            img_resized.save(out_path)

            print(f"Saved: {out_path}")

# 137,268개 생성