import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, DataLoader


class SimpleImageFolder(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        try:
            x = Image.open(image_path).convert("RGB")
        except:
            return None
        if self.transform is not None:
            x = self.transform(x)
        return x, image_path

    def __len__(self):
        return len(self.image_paths)

class Pipeline:
    def __init__(self, config, model, transform, tokenizer, device):
        self.config = config
        self.model = model.to(device).eval()
        self.transform = transform
        self.tokenizer = tokenizer
        self.device = device

        # config 기반 설정
        self.top_k = config.get("top_k", 5)
        self.metric = config.get("retrieval_metric", "cosine")
        self.batch_size = config.get("encoding", {}).get("batch_size", 32)
        self.num_workers = config.get("encoding", {}).get("num_workers", 1)

        self.image_paths = []
        self.image_features = None

    def encode_query(self, sketch: Image.Image, caption: str) -> torch.Tensor:
        sketch_tensor = self.transform(sketch).unsqueeze(0).to(self.device)
        text_tensor = self.tokenizer(caption).to(self.device)

        with torch.no_grad():
            sketch_feat = self.model.encode_sketch(sketch_tensor)
            text_feat = self.model.encode_text(text_tensor)
            sketch_feat = sketch_feat / sketch_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            fused_feat = self.model.feature_fuse(sketch_feat, text_feat)
        return fused_feat

    def index_database(self, image_paths: list[str]):
        self.image_paths = image_paths

        dataset = SimpleImageFolder(image_paths, transform=self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False,
        )

        features = []
        all_paths = []
        with torch.no_grad():
            print(f"[INFO] Start encoding")
            for batch_num, batch in enumerate(dataloader):
                print(f"Batch {batch_num}", end='')
                imgs, paths = batch
                imgs = imgs.to(self.device)

                feats = self.encode_image(imgs)
                feats = feats / feats.norm(dim=-1, keepdim=True)

                features.append(feats.cpu())
                all_paths.extend(paths)
                print(" -- Done")

        self.image_paths = all_paths
        self.image_features = torch.cat(features, dim=0).numpy()

        if self.metric == 'cosine':
            self.nbrs = NearestNeighbors(n_neighbors=self.top_k, metric='cosine').fit(self.image_features)

        print(f"[INFO] Indexed {len(self.image_paths)} images.")

    def retrieve(self, query_feature: torch.Tensor) -> list[str]:
        query = query_feature.squeeze(0).cpu().numpy()
        if self.metric == "cosine":
            _, indices = self.nbrs.kneighbors(query)
        elif self.metric == "dot":
            scores = np.dot(self.image_features, query)
            indices = np.argsort(-scores)[:self.top_k]
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        return [self.image_paths[i] for i in indices[0]] if self.metric == "cosine" else [self.image_paths[i] for i in indices]

    def run_retrieval(self, sketch_path: str, caption: str) -> list[str]:
        sketch = Image.open(sketch_path).convert("RGB")
        query_feat = self.encode_query(sketch, caption)
        return self.retrieve(query_feat)
    
    def __call__(self, images, sketches, captions):
        return self.model.foward(images, sketches, captions)

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)