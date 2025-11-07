from PIL import Image, ImageDraw
import pytesseract
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
import torchvision.transforms as T

# ============ 1. 生成一张简单的测试图 ============
img = Image.new("RGB", (512, 256), (255, 255, 255))
d = ImageDraw.Draw(img)
d.text((20, 100), "When Monday hits... stay positive!", fill=(0, 0, 0))
img_path = "data/samples/test_meme.jpg"
img.save(img_path)

# ============ 2. OCR 测试 ============
text = pytesseract.image_to_string(Image.open(img_path))
print("[OCR 输出]：", text.strip())

# ============ 3. BERT 文本特征提取 ============
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = AutoModel.from_pretrained("bert-base-uncased").eval()
inputs = tok(text, return_tensors="pt", max_length=42, truncation=True)
with torch.no_grad():
    pooled = bert(**inputs).pooler_output  # [1, 768]
text_feat = nn.Sequential(nn.Linear(768, 256), nn.ReLU())(pooled)
print("[BERT 特征维度]：", tuple(text_feat.shape))

# ============ 4. VGG19 图像特征提取 ============
device = "cuda" if torch.cuda.is_available() else "cpu"
weights = VGG19_Weights.IMAGENET1K_V1
vgg = vgg19(weights=weights).eval().to(device)
vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])  # 输出4096维
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])
im = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
with torch.no_grad():
    v = vgg(im)  # [1, 4096]
img_feat = nn.Sequential(nn.Linear(4096, 256), nn.ReLU()).to(device)(v)
print("[VGG19 特征维度]：", tuple(img_feat.shape))

# ============ 5. 融合 + 分类层测试 ============
fusion = torch.cat([text_feat.to(device), img_feat], dim=1)  # [1, 512]
head = nn.Sequential(
    nn.Linear(512, 64),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, 3)
).to(device)

with torch.no_grad():
    logits = head(fusion)
    probs = torch.softmax(logits, dim=-1)
print("[融合层输出概率]：", probs.cpu().numpy())

print("\n✅ Smoke test 成功！环境和模型结构均正常。\n")