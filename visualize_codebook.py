import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from taming.models.vqgan import VQModel
from omegaconf import OmegaConf
import argparse

def load_model(config_path, ckpt_path):
    config = OmegaConf.load(config_path)
    model = VQModel(**config.model.params)
    model.init_from_ckpt(ckpt_path)
    model.eval()
    return model

def visualize_codebook(model, save_dir, grid_size=8):
    os.makedirs(save_dir, exist_ok=True)
    
    # Get codebook embeddings
    codebook = model.quantize.embedding.weight.detach().cpu()
    n_embeddings = codebook.shape[0]
    embed_dim = codebook.shape[1]
    
    print(f"Codebook shape: {codebook.shape}")
    print(f"Number of embeddings: {n_embeddings}")
    print(f"Embedding dimension: {embed_dim}")
    
    # 임베딩 벡터를 이미지 형태로 변환
    patch_size = 16
    try:
        codebook = codebook.view(n_embeddings, 1, patch_size, patch_size)
        print(f"Reshaped codebook shape: {codebook.shape}")
    except RuntimeError as e:
        print(f"Error reshaping codebook: {e}")
        return
    
    # Normalize to [0, 1] range
    codebook = (codebook - codebook.min()) / (codebook.max() - codebook.min())
    
    # 이미지 시각화
    n_rows = n_cols = grid_size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
    
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx < n_embeddings:
                img = codebook[idx].squeeze().numpy()
                axes[i, j].imshow(img, cmap='gray')
                axes[i, j].axis('off')
            else:
                axes[i, j].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'codebook_visualization.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Visualization saved to {save_path}")
    
    # 숫자 값 시각화
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
    
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx < n_embeddings:
                values = codebook[idx].squeeze().numpy()
                # 16x16 그리드에 숫자 표시
                axes[i, j].set_xticks(np.arange(16))
                axes[i, j].set_yticks(np.arange(16))
                axes[i, j].set_xticklabels([])
                axes[i, j].set_yticklabels([])
                
                # 각 셀에 숫자 표시 (소수점 2자리까지)
                for x in range(16):
                    for y in range(16):
                        axes[i, j].text(x, y, f'{values[y, x]:.2f}', 
                                      ha='center', va='center', 
                                      fontsize=4, color='black')
                
                axes[i, j].set_title(f'Code {idx}', fontsize=8)
                axes[i, j].grid(True)
            else:
                axes[i, j].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'codebook_values.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Codebook values saved to {save_path}")
    
    # 통계 정보 저장
    stats = {
        'mean': codebook.mean().item(),
        'std': codebook.std().item(),
        'min': codebook.min().item(),
        'max': codebook.max().item(),
        'per_codebook_stats': []
    }
    
    for i in range(min(n_embeddings, grid_size * grid_size)):
        code_stats = {
            'code_idx': i,
            'mean': codebook[i].mean().item(),
            'std': codebook[i].std().item(),
            'min': codebook[i].min().item(),
            'max': codebook[i].max().item()
        }
        stats['per_codebook_stats'].append(code_stats)
    
    # 통계 정보를 텍스트 파일로 저장
    with open(os.path.join(save_dir, 'codebook_stats.txt'), 'w') as f:
        f.write("전체 Codebook 통계:\n")
        f.write(f"평균: {stats['mean']:.4f}\n")
        f.write(f"표준편차: {stats['std']:.4f}\n")
        f.write(f"최소값: {stats['min']:.4f}\n")
        f.write(f"최대값: {stats['max']:.4f}\n\n")
        
        f.write("개별 Codebook 항목 통계:\n")
        for code_stat in stats['per_codebook_stats']:
            f.write(f"\nCode {code_stat['code_idx']}:\n")
            f.write(f"평균: {code_stat['mean']:.4f}\n")
            f.write(f"표준편차: {code_stat['std']:.4f}\n")
            f.write(f"최소값: {code_stat['min']:.4f}\n")
            f.write(f"최대값: {code_stat['max']:.4f}\n")
    
    print(f"Codebook statistics saved to {os.path.join(save_dir, 'codebook_stats.txt')}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--save_dir', type=str, default='codebook_visualization', help='Directory to save visualizations')
    args = parser.parse_args()
    
    print(f"Loading model from config: {args.config}")
    print(f"Checkpoint path: {args.ckpt}")
    
    model = load_model(args.config, args.ckpt)
    print("Model loaded successfully")
    
    visualize_codebook(model, args.save_dir)
    print("Visualization completed")

if __name__ == '__main__':
    main() 
    
    
"""
python visualize_codebook.py --config configs/brats_vqgan_2.yaml --ckpt logs/exp2_140000/checkpoints/last.ckpt --save_dir codebook_visualization
"""