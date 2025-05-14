import os
import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms
from omegaconf import OmegaConf
import sys
import random
from pathlib import Path
import shutil
import numpy as np
import albumentations
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.filters import threshold_otsu
sys.path.append(os.getcwd())

def load_model_from_config(config, ckpt):
    from taming.models.vqgan import VQModel
    model = VQModel(**config.model.params)
    model.init_from_ckpt(ckpt)
    return model

def load_config(config_path):
    config = OmegaConf.load(config_path)
    return config

def get_random_images(data_dir, num_images=100):
    # 모든 이미지 파일 경로 수집
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg']:
        image_files.extend(list(Path(data_dir).rglob(f'*{ext}')))
    
    # 무작위로 num_images개 선택
    selected_files = random.sample(image_files, min(num_images, len(image_files)))
    return selected_files

def preprocess_image(image_path, size=256):
    # VQGAN_BraTs의 base.py의 전처리 방식 적용
    image = Image.open(image_path)
    image = image.convert("L")  # 흑백 처리
    image = np.array(image).astype(np.uint8)
    
    # albumentations 전처리
    rescaler = albumentations.SmallestMaxSize(max_size=size)
    cropper = albumentations.CenterCrop(height=size, width=size)
    preprocessor = albumentations.Compose([rescaler, cropper])
    image = preprocessor(image=image)["image"]
    
    # 정규화
    image = (image/127.5 - 1.0).astype(np.float32)
    image = image[..., np.newaxis]  # 채널 차원 추가
    
    # 텐서로 변환
    image = torch.from_numpy(image).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    return image

def save_image(x, path):
    c, h, w = x.shape
    assert c == 1  # grayscale 이미지 확인
    x = x.clone()
    x = x.detach().cpu().numpy()
    x = x.squeeze(0)  # (H, W)
    
    # -1~1 범위를 0~255로 변환
    x = ((x + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    
    # 0~255 범위에서 배경 노이즈 제거 (약 5 정도의 값)
    x[x < 5] = 0
    
    Image.fromarray(x, mode='L').save(path)

def list_image_files(folder):
    exts = ['.png', '.jpg', '.jpeg', '.bmp']
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if os.path.splitext(f)[-1].lower() in exts]
    files.sort()
    return files

def calculate_ssim_psnr(input_dir, recon_dir, original_names):
    input_files = list_image_files(input_dir)
    recon_files = list_image_files(recon_dir)
    input_files.sort()
    recon_files.sort()
    
    if len(input_files) != len(recon_files):
        print(f"Warning: Number of input and recon images do not match: {len(input_files)} vs {len(recon_files)}")
    
    # Contrast별로 결과를 저장할 딕셔너리
    contrast_scores = {
        'T1': {'ssim': [], 'psnr': []},
        'T2': {'ssim': [], 'psnr': []},
        'T1gd': {'ssim': [], 'psnr': []},
        'FLAIR': {'ssim': [], 'psnr': []}
    }
    
    n = min(len(input_files), len(recon_files))
    for i in range(n):
        input_path = input_files[i]
        recon_path = recon_files[i]
        
        # 원본 파일 이름에서 Contrast 정보 추출
        input_filename = os.path.basename(input_path)
        original_name = original_names[input_filename]
        contrast = None
        if '_t1.png' in original_name.lower():
            contrast = 'T1'
        elif '_t2.png' in original_name.lower():
            contrast = 'T2'
        elif '_t1gd.png' in original_name.lower():
            contrast = 'T1gd'
        elif '_flair.png' in original_name.lower():
            contrast = 'FLAIR'
        
        if contrast:
            # 이미지 로드
            img1 = np.array(Image.open(input_path).convert('L'))
            img2 = np.array(Image.open(recon_path).convert('L'))
            
            # SSIM 계산
            ssim_score = ssim(img1, img2, data_range=255)
            
            # PSNR 계산
            mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
            if mse == 0:
                psnr_score = float('inf')
            else:
                psnr_score = 20 * np.log10(255.0 / np.sqrt(mse))
            
            contrast_scores[contrast]['ssim'].append(ssim_score)
            contrast_scores[contrast]['psnr'].append(psnr_score)
    
    # 각 Contrast별 평균 계산
    results = {}
    for contrast, scores in contrast_scores.items():
        if scores['ssim']:  # 해당 Contrast의 이미지가 있는 경우에만
            results[contrast] = {
                'mean_ssim': np.mean(scores['ssim']),
                'mean_psnr': np.mean(scores['psnr']),
                'count': len(scores['ssim'])
            }
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/brats_vqgan_1.yaml',
                      help='path to config file')
    parser.add_argument('--ckpt', type=str, default='logs/experiement_1/checkpoints/last.ckpt',
                      help='path to checkpoint file')
    parser.add_argument('--data_dir', type=str, default='/mnt/MW/VQGAN_BraTs/data/BraTs2020',
                      help='path to data directory')
    parser.add_argument('--output_dir', type=str, default='/mnt/MW/VQGAN_BraTs',
                      help='base directory to save reconstructed images')
    parser.add_argument('--num_images', type=int, default=100,
                      help='number of images to reconstruct')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # yaml 파일 이름에서 숫자 추출
    yaml_name = os.path.basename(args.config)
    yaml_num = yaml_name.split('_')[-1].split('.')[0]  # brats_vqgan_1.yaml -> 1
    
    # reconstructions 폴더 이름 설정
    reconstructions_dir = os.path.join(args.output_dir, f'reconstructions_{yaml_num}')
    
    # 입력/출력 디렉토리 생성
    input_dir = os.path.join(reconstructions_dir, 'input')
    recon_dir = os.path.join(reconstructions_dir, 'recon')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)

    # Load config and model
    config = load_config(args.config)
    model = load_model_from_config(config, args.ckpt)
    model = model.to(args.device)
    model.eval()
    
    # 무작위 이미지 선택
    image_files = get_random_images(args.data_dir, args.num_images)
    print(f"Selected {len(image_files)} images for reconstruction")

    # 원본 파일 이름과 Contrast 정보를 저장할 딕셔너리
    original_names = {}

    # 각 이미지에 대해 reconstruction 수행
    for i, img_path in enumerate(image_files):
        try:
            # 원본 파일 이름 저장
            original_name = os.path.basename(img_path)
            input_filename = f'input_{i:03d}.png'
            original_names[input_filename] = original_name

            # 원본 이미지 복사
            input_path = os.path.join(input_dir, input_filename)
            shutil.copy2(img_path, input_path)

            # 이미지 전처리
            img_tensor = preprocess_image(img_path).unsqueeze(0).to(args.device)

            # Reconstruction
            with torch.no_grad():
                xrec, _ = model(img_tensor)
            
            # 결과 저장
            xrec = xrec.squeeze(0).cpu()  # 배치 차원 제거
            save_image(xrec, os.path.join(recon_dir, f'recon_{i:03d}.png'))
            
            print(f"Processed {i+1}/{len(image_files)}: {img_path.name}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue

    print(f"Reconstruction completed.")
    print(f"Original images saved in: {input_dir}")
    print(f"Reconstructed images saved in: {recon_dir}")

    # SSIM/PSNR 계산 및 저장
    print(f"Calculating SSIM and PSNR between {input_dir} and {recon_dir}")
    contrast_results = calculate_ssim_psnr(input_dir, recon_dir, original_names)
    
    # 결과 출력
    print("\nResults by Contrast:")
    for contrast, scores in contrast_results.items():
        print(f"\n{contrast}:")
        print(f"Number of images: {scores['count']}")
        print(f"Mean SSIM: {scores['mean_ssim']:.4f}")
        print(f"Mean PSNR: {scores['mean_psnr']:.4f}")
    
    # 결과 파일 저장
    result_file = os.path.join(reconstructions_dir, 'ssim_psnr_score.txt')
    with open(result_file, 'w') as f:
        f.write("Results by Contrast:\n")
        for contrast, scores in contrast_results.items():
            f.write(f"\n{contrast}:\n")
            f.write(f"Number of images: {scores['count']}\n")
            f.write(f"Mean SSIM: {scores['mean_ssim']:.4f}\n")
            f.write(f"Mean PSNR: {scores['mean_psnr']:.4f}\n")
        f.write(f"\nInput Directory: {input_dir}\n")
        f.write(f"Reconstruction Directory: {recon_dir}\n")
    print(f"\nScores saved to: {result_file}")

if __name__ == "__main__":
    main()

"""
python reconstruct_ssim_psnr.py \
--config configs/brats_vqgan_1.yaml \
--ckpt logs/exp1_150000/checkpoints/last.ckpt \
--data_dir /mnt/MW/VQGAN_BraTs2020/data/BraTs2020 \
--output_dir /mnt/MW/VQGAN_BraTs2020 \
--num_images 500
"""
