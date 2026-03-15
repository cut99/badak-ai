import sys
from models.florence_model import FlorenceModel
from models.translation_model import TranslationModel
from PIL import Image

image_path = "../sampleimage/im10.jpeg"

try:
    img = Image.open(image_path).convert("RGB")
except Exception as e:
    print(f"Please provide an image named 'tes.jpg'. Error: {e}")
    sys.exit(1)

translation = TranslationModel()
model = FlorenceModel(translation_model=translation)
print("=== OD ===")
res_od = model.run_task(img, "<OD>")
print(res_od)
print("=== REGION_PROPOSAL ===")
res_rp = model.run_task(img, "<REGION_PROPOSAL>")
print(res_rp)
print("=== DENSE_REGION_CAPTION ===")
res_drc = model.run_task(img, "<DENSE_REGION_CAPTION>")
print(res_drc)

# Get current tags function
print("=== CURRENT get_tags ===")
tags = model.get_tags(img, top_k=20)
print(tags)
