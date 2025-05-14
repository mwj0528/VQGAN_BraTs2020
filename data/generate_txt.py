import os
import random

# 저장된 이미지 경로
output_root = "/mnt/MW/VQGAN_BraTs/data/BraTs2020"

# 전체 이미지 경로 리스트
all_paths = []

# .png 파일들을 모두 찾음
for file in os.listdir(output_root):
    if file.endswith(".png"):  # PNG 파일만 선택
        img_path = os.path.join(output_root, file)
        all_paths.append(img_path)

# 전체 이미지 수
total_images = len(all_paths)
print(f"Total images: {total_images}")

# 훈련 데이터와 검증 데이터 비율 (80:20)
train_size = int(0.8 * total_images)
val_size = total_images - train_size

# 데이터 섞기
random.shuffle(all_paths)  # 경로를 랜덤하게 섞기

# 훈련 데이터와 검증 데이터 분리
train_paths = all_paths[:train_size]  # 훈련 데이터 (80%)
val_paths = all_paths[train_size:]    # 검증 데이터 (20%)

# brats_train.txt 파일 작성
train_txt_path = "/mnt/MW/VQGAN_BraTs/data/brats_train.txt"
with open(train_txt_path, 'w') as f:
    for path in train_paths:
        # 경로 쓰기
        f.write(path + "\n")
        
print(f"Training file written at {train_txt_path}")

# brats_val.txt 파일 작성
val_txt_path = "/mnt/MW/VQGAN_BraTs/data/brats_val.txt"
with open(val_txt_path, 'w') as f:
    for path in val_paths:
        # 경로 쓰기
        f.write(path + "\n")

print(f"Validation file written at {val_txt_path}")
