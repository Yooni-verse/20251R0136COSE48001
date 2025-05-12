from pathlib import Path
import sys

CODE_PATH = Path('./codes/')
sys.path.append(str(CODE_PATH))

import torch
import yaml
import json
from PIL import Image
from model.pipeline import Pipeline
from clip.model import convert_weights, CLIP
from clip.clip import _transform, load, tokenize

def get_concat_hn(ims):
    sum_w = 0
    for im in ims:
        #im = im.resize((256,256))
        sum_w += 256#im.width
        
    max_h = 256#max([im.height for im in ims])
    
    dst = Image.new('RGB', (sum_w ,max_h))
    cur_x = 0
    for im in ims:
        dst.paste(im.resize((256,256)), (cur_x, 0))
        cur_x += 256#im.width
    #dst.paste(im2, (im1.width, 0))
    #dst.paste(im3, (im1.width+im2.width, 0))
    return dst

def run(config):
    # Set device
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load model
    with open(config["model_config"], 'r') as f:
        model_info = json.load(f)

    model = CLIP(**model_info)

    checkpoint = torch.load(config["model_ckpt"], weights_only=False, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if next(iter(state_dict)).startswith("module."):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    convert_weights(model)

    # Load transform and tokenizer
    transform = _transform(model.visual.input_resolution, is_train=config["is_train"])
    tokenizer = tokenize

    for name, module in model.named_modules():
        if not module.requires_grad_:
            print(f"name: {name}, module: {module}")
    return

    # Load pipeline
    pipeline = Pipeline(
        config=config,
        model=model,
        transform=transform,
        tokenizer=tokenizer,
        device=device
    )

    # Prepare images and caption
    image_dir = config["encoding"]["image_dir"]
    sketch_path = config["inference"]["sketch_path"]
    caption = config["inference"]["caption"]

    image_paths = sorted([str(p) for p in Path(image_dir).glob("*") if p.suffix.lower() in [".jpg", ".png"]])
    
    # Inference
    pipeline.index_database(image_paths)
    result_paths = pipeline.run_retrieval(sketch_path, image_paths)

    # Results
    print(f"\n[QUERY] Caption: {caption}")
    print("[RESULT] Top-K retrieved image paths:")
    for p in result_paths:
        print(" -", p)

    # Save result image
    sketch = Image.open(sketch_path)
    im_list = [sketch] + [Image.open(p) for p in result_paths]
    image = get_concat_hn(im_list)
    save_path = "retrieval_result.jpg"
    image.save(save_path)
    print(f"🔍 Retrieval result saved to: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    run(config)
