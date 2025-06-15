import sys
from pathlib import Path

CODE_PATH = Path('./codes/')
sys.path.append(str(CODE_PATH))

from flask import Flask, request, jsonify, render_template
import base64, io, uuid, os, json
from PIL import Image

import torch, yaml
from codes.model.pipeline import Pipeline
from codes.clip.model import CLIP
from codes.clip.clip import _transform, tokenize

cfg_path = 'configs/config.yaml'
config = yaml.safe_load(open(cfg_path))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load model
with open(config["model_config"]) as f:
    info = json.load(f)
model = CLIP(**info)
checkpoint = torch.load(config["model_ckpt"], map_location=device, weights_only=False)
state = checkpoint.get("state_dict", checkpoint)
state = {k.replace("module.", "", 1): v for k, v in state.items()}
model.load_state_dict(state, strict=False)
model.to(device)
transform = _transform(model.visual.input_resolution, is_train=False)
pipeline = Pipeline(config, model, transform, tokenize, device)

# DB indexing
img_dir = config["encoding"]["image_dir"]
img_paths = sorted(str(p) for p in Path(img_dir).glob("*") if p.suffix.lower() in [".jpg", ".png"])
pipeline.index_database(img_paths)

# IKEA 메타데이터 로딩
with open("ikea_database/ikea_product_info.json", encoding="utf-8") as f:
    ikea_info = json.load(f)

# Flask
app = Flask(__name__, static_folder='static', template_folder='templates')
UPLOAD_DIR = Path('uploads')
UPLOAD_DIR.mkdir(exist_ok=True)

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/infer', methods=['POST'])
def infer():
    data = request.get_json()
    sketch_b64 = data['sketch'].split(',')[1]
    caption = data.get('caption', '')

    # Decode image
    im = Image.open(io.BytesIO(base64.b64decode(sketch_b64))).convert('RGB')
    tmp_name = UPLOAD_DIR / f"{uuid.uuid4().hex}.png"
    im.save(tmp_name)

    # 검색
    result_paths = pipeline.run_retrieval(str(tmp_name), caption)

    results = []
    for path in result_paths:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        image_b64 = "data:image/jpeg;base64," + b64

        # 키 추출: filename → 'sofa-10489009-0.png' → 'sofa-10489009'
        key = "-".join(Path(path).stem.split("-")[:-1])
        meta = ikea_info.get(key, {})

        results.append({
            "image": image_b64,
            "name": meta.get("name", "Unknown"),
            "description": meta.get("description", ""),
            "price": meta.get("price", "-"),
            "rating": meta.get("rating", "-"),
            "num_reviews": meta.get("num_reviews", "-"),
            "link": meta.get("link", "#")
        })

    tmp_name.unlink(missing_ok=True)
    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
