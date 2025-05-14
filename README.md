# VQGAN_BraTs2020
Using VQGAN to perform reconstruction task using multiple modality data of 2D slice of BraTs2020

## Reference
### [taming-transformers](https://github.com/CompVis/taming-transformers.git)

## Installation
Implementation is conducted on Python 3.10. To install the environment, please run the following.
```
conda env create -f environment.yaml
conda activate taming and pip install -e .
put your .jpg files in a folder your_folder
create 2 text files a xx_train.txt and xx_test.txt that point to the files in your training and test set respectively (for example find $(pwd)/your_folder -name "*.jpg" > train.txt)
adapt configs/custom_vqgan.yaml to point to these 2 files
```
## Run

I use a 2 NVIDIA TITAN RTX GPU for our experiments.

### Train
```
python main.py --base configs/brats_vqgan_x.yaml -t True --gpus 0,1 to train on two GPUs. Use --gpus 0, (with a trailing comma) to train on a single GPU.
```
### Sample
```
python reconstruct_ssim_psnr.py \
--config configs/brats_vqgan_1.yaml \
--ckpt logs/experiement_1/checkpoints/last.ckpt \
--data_dir /mnt/MW/VQGAN_BraTs/data/BraTs2020 \
--output_dir /mnt/MW/VQGAN_BraTs \
--num_images 500
```

## Example Result

### downsampling factor f = 16, discriminator start = 50001(brats_vqgan_1.yaml)
![image](https://github.com/user-attachments/assets/d2929652-3289-400a-af87-3f1fc3832d7a)


### downsampling factor f = 8, discriminator start = 50001(brats_vqgan_2.yaml)
![image](https://github.com/user-attachments/assets/7b4cfd8f-e77d-4caf-ad13-2d2d17ecafa0)
