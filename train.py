import yaml
import torch
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from clip.clip import tokenize
from pathlib import Path
from PIL import Image
from codes.model.pipeline import Pipeline
from codes.model_loader import load_model_and_transform

# ===== 학습용 Dataset 정의 =====
class SketchImageCaptionDataset(Dataset):
    def __init__(self, sketch_dir, image_dir, annotation_file, transform):
        with open(annotation_file, 'r') as f:
            self.annotations = yaml.safe_load(f)  # 또는 json.load
        self.sketch_dir = Path(sketch_dir)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]
        sketch_path = self.sketch_dir / item['sketch']
        image_path = self.image_dir / item['image']
        caption = item['caption']

        sketch = Image.open(sketch_path).convert("RGB")
        image = Image.open(image_path).convert("RGB")

        sketch_tensor = self.transform(sketch)
        image_tensor = self.transform(image)
        caption_tensor = tokenize([caption])[0]

        return sketch_tensor, image_tensor, caption_tensor

# ===== Contrastive Loss =====
def contrastive_loss(sketch_feats, image_feats, temperature=0.07):
    sketch_feats = sketch_feats / sketch_feats.norm(dim=-1, keepdim=True)
    image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
    logits = torch.matmul(sketch_feats, image_feats.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return torch.nn.functional.cross_entropy(logits, labels)

# ===== Train Loop =====
def train():
    # === config 로드 ===
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model, transform = load_model_and_transform(
        config["model_config"],
        config["model_weights"],
        config.get("transform_type", "train"),
        device
    )

    pipeline = Pipeline(config=config, model=model, transform=transform, device=device)

    # === Dataset 구성 ===
    train_cfg = config["train"]
    dataset = SketchImageCaptionDataset(
        sketch_dir=train_cfg["dataset"]["sketch_folder"],
        image_dir=train_cfg["dataset"]["image_folder"],
        annotation_file=train_cfg["dataset"]["annotation_file"],
        transform=transform
    )
    dataloader = DataLoader(dataset,
                            batch_size=train_cfg["batch_size"],
                            shuffle=True,
                            num_workers=train_cfg["num_workers"],
                            drop_last=True)

    # === Optimizer 정의 ===
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    temperature = train_cfg.get("temperature", 0.07)
    save_path = Path(train_cfg.get("save_path", "checkpoints"))
    save_path.mkdir(parents=True, exist_ok=True)

    # === 학습 루프 ===
    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for sketch, image, captions in tqdm(dataloader, desc=f"Epoch {epoch}"):
            sketch = sketch.to(device)
            image = image.to(device)

            sketch_feats = model.encode_sketch(sketch)
            image_feats = model.encode_image(image)

            loss = contrastive_loss(sketch_feats, image_feats, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

        # === 모델 저장 ===
        if epoch % train_cfg.get("save_every", 10) == 0 or epoch == train_cfg["epochs"]:
            ckpt_path = save_path / f"tsbir_epoch{epoch}.pt"
            torch.save({'state_dict': model.state_dict()}, ckpt_path)
            print(f"✅ Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    train()
