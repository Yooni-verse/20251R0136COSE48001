from datasets import load_dataset
import time
import os

save_dir = './image'
os.makedirs(save_dir, exist_ok=True)

ds = load_dataset("Arkan0ID/furniture-dataset")
ds = ds['train']

for i, data in enumerate(ds):
    image = data['image']
    image.save(f"{save_dir}/{i}.png")
    time.sleep(0.5)  # 0.5초 대기 (또는 1초)