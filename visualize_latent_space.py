import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from taming.models.vqgan import VQModel
from omegaconf import OmegaConf
import argparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from taming.data.custom import CustomTest
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import time

def wait_for_gpu():
    """GPU 작업이 완료될 때까지 대기"""
    torch.cuda.synchronize()

def load_model(config_path, ckpt_path):
    config = OmegaConf.load(config_path)
    model = VQModel(**config.model.params)
    model.init_from_ckpt(ckpt_path)
    model.eval()
    return model

def get_latent_vectors(model, dataloader, dataset, device='cuda:0'):
    model = model.to(device)
    latent_vectors = []
    image_paths = []
    idx = 0

    # dataset이 Subset인 경우 원본 데이터셋의 data 속성에 접근
    if isinstance(dataset, torch.utils.data.Subset):
        data_obj = dataset.dataset.data
    else:
        data_obj = dataset.data

    # ImagePaths 객체에서 경로 리스트 가져오기
    if hasattr(data_obj, 'labels') and 'file_path_' in data_obj.labels:
        path_list = data_obj.labels['file_path_']
    else:
        raise AttributeError(f"데이터셋에서 경로 리스트를 찾을 수 없습니다: {dir(data_obj)}")

    print(f"path_list 길이: {len(path_list)}")

    # Subset인 경우 인덱스에 맞는 경로만 선택
    if isinstance(dataset, torch.utils.data.Subset):
        try:
            path_list = [path_list[i] for i in dataset.indices]
        except IndexError as e:
            print(f"인덱스 오류 발생: path_list 길이={len(path_list)}, 최대 인덱스={max(dataset.indices)}")
            raise e

    print("Latent vector 추출 중...")
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="배치 처리", position=0, leave=True)
        for batch in pbar:
            torch.cuda.synchronize()  # GPU 대기
            
            x = batch['image']
            if x.ndim == 4 and x.shape[-1] == 1:
                x = x.permute(0, 3, 1, 2)
            x = x.to(device)
            
            h = model.encoder(x)
            h = model.quant_conv(h)
            quant, _, info = model.quantize(h)
                
            latent_vectors.append(quant.cpu().numpy())
            batch_size = x.shape[0]
            image_paths.extend(path_list[idx:idx+batch_size])
            idx += batch_size
            
            # 진행 상황 업데이트
            pbar.set_postfix({'처리된 이미지': idx})
        
        pbar.close()

    return np.concatenate(latent_vectors, axis=0), image_paths

def get_image_type(path):
    # 파일명에서 이미지 타입 추출 (t1, t1gd, t2, flair)
    filename = os.path.basename(path)
    if 't1gd' in filename:
        return 't1gd'
    elif 't1' in filename:
        return 't1'
    elif 't2' in filename:
        return 't2'
    elif 'flair' in filename:
        return 'flair'
    return 'unknown'

def visualize_latent_space(latent_vectors, image_paths, save_dir, use_tsne=False, sample_size=1000):
    os.makedirs(save_dir, exist_ok=True)
    
    # 차원 축소
    if use_tsne:
        print("t-SNE 차원 축소 수행 중...")
        reducer = TSNE(n_components=2, random_state=42)
        title = 't-SNE Visualization of Latent Space'
    else:
        print("PCA 차원 축소 수행 중...")
        reducer = PCA(n_components=2)
        title = 'PCA Visualization of Latent Space'
    
    print("차원 축소 진행 중...")
    latent_2d = reducer.fit_transform(latent_vectors.reshape(len(latent_vectors), -1))
    
    # 이미지 타입별로 분류
    print("이미지 타입 분류 중...")
    image_types = [get_image_type(path) for path in tqdm(image_paths, desc="이미지 타입 분류")]
    unique_types = list(set(image_types))
    
    # 타입별로 다른 색상 사용
    colors = sns.color_palette('husl', n_colors=len(unique_types))
    type_colors = dict(zip(unique_types, colors))
    
    # 타입별로 시각화
    print("시각화 생성 중...")
    plt.figure(figsize=(12, 10))
    for img_type in tqdm(unique_types, desc="타입별 시각화"):
        mask = [t == img_type for t in image_types]
        plt.scatter(latent_2d[mask, 0], latent_2d[mask, 1], 
                   label=img_type, alpha=0.5, color=type_colors[img_type])
    
    plt.title(f'{title} by Image Type')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'latent_space_{"tsne" if use_tsne else "pca"}.png'))
    plt.close()
    
    # 유사도 분석
    print("유사도 행렬 계산 중...")
    similarity_matrix = squareform(pdist(latent_2d))
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap='viridis')
    plt.title('Latent Space Similarity Matrix')
    plt.savefig(os.path.join(save_dir, f'latent_space_similarity_{"tsne" if use_tsne else "pca"}.png'))
    plt.close()
    
    # 타입별 통계 정보 저장
    print("통계 정보 계산 중...")
    stats = {}
    for img_type in tqdm(unique_types, desc="통계 계산"):
        mask = [t == img_type for t in image_types]
        type_vectors = latent_2d[mask]
        stats[img_type] = {
            'mean': np.mean(type_vectors, axis=0),
            'std': np.std(type_vectors, axis=0),
            'count': np.sum(mask)
        }
    
    # 통계 정보를 텍스트 파일로 저장
    print("결과 저장 중...")
    with open(os.path.join(save_dir, f'latent_space_stats_{"tsne" if use_tsne else "pca"}.txt'), 'w') as f:
        method = "t-SNE" if use_tsne else "PCA"
        f.write(f"Latent Space Analysis ({method})\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"전체 데이터 수: {len(latent_vectors)}개\n")
        f.write(f"시각화에 사용된 데이터 수: {sample_size}개\n\n")
        
        for img_type, stat in stats.items():
            f.write(f"Image Type: {img_type}\n")
            f.write(f"Number of samples: {stat['count']}\n")
            f.write(f"Mean position: {stat['mean']}\n")
            f.write(f"Standard deviation: {stat['std']}\n")
            f.write("-" * 30 + "\n")
    
    print("완료!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to validation data')
    parser.add_argument('--save_dir', type=str, default='latent_space_analysis', help='Directory to save visualizations')
    parser.add_argument('--use_tsne', action='store_true', help='Use t-SNE instead of PCA for dimensionality reduction')
    parser.add_argument('--sample_size', type=int, default=5000, help='Number of samples to use for visualization')
    args = parser.parse_args()
    
    # 시작 전에 GPU 작업 완료 대기
    wait_for_gpu()
    
    # 모델 로드
    model = load_model(args.config, args.ckpt)
    
    # 데이터셋 생성 및 샘플링
    full_dataset = CustomTest(size=256, test_images_list_file='/mnt/MW/VQGAN_BraTs/data/brats_val.txt')
    
    # 전체 데이터셋에서 지정된 크기만큼 랜덤 샘플링
    if len(full_dataset) > args.sample_size:
        indices = np.random.choice(len(full_dataset), args.sample_size, replace=False)
        dataset = torch.utils.data.Subset(full_dataset, indices)
        print(f"전체 데이터: {len(full_dataset)}개 중 {args.sample_size}개 샘플링")
    else:
        dataset = full_dataset
        print(f"전체 데이터: {len(full_dataset)}개 사용")
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # latent vector 추출
    latent_vectors, image_paths = get_latent_vectors(model, dataloader, dataset)
    
    # 시각화
    visualize_latent_space(latent_vectors, image_paths, args.save_dir, args.use_tsne, args.sample_size)

if __name__ == '__main__':
    main() 
    
"""

python visualize_latent_space.py --config configs/brats_vqgan_2.yaml --ckpt logs/exp2_140000/checkpoints/last.ckpt --data_dir /mnt/MW/VQGAN_BraTs/data/BraTs2020 --save_dir latent_space_analysis

python visualize_latent_space.py --config configs/brats_vqgan_2.yaml --ckpt logs/exp2_140000/checkpoints/last.ckpt --data_dir /mnt/MW/VQGAN_BraTs/data/BraTs2020 --save_dir latent_space_analysis --use_tsne

"""