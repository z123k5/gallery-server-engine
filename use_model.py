import io
from PIL import Image
import requests
import clip
import torch
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from torchsummary import summary

import os
os.environ["HF_HOME"] = "./"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
proxy_dict = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}

query_texts = ['人']  # 这里是输入文本的，可以随意替换。
# query_texts = ['啤酒','晚宴','合影','旅游']  # 这里是输入文本的，可以随意替换。
# 加载Taiyi 中文 text encoder
text_tokenizer = BertTokenizer.from_pretrained(
    "./Taiyi-CLIP-Roberta-large-326M-Chinese")
text_encoder = BertForSequenceClassification.from_pretrained(
    "./Taiyi-CLIP-Roberta-large-326M-Chinese").eval()
text = text_tokenizer(query_texts, return_tensors='pt', padding=True)['input_ids']

url = [
    "C:/Users/Kang/Desktop/DCIM/temp/_DSC2035.JPG",
    "C:/Users/Kang/Desktop/DCIM/temp/_DSC2056.JPG",
    "C:/Users/Kang/Desktop/DCIM/temp/_DSC2807.JPG",
    "C:/Users/Kang/Desktop/DCIM/temp/_DSC2808.JPG",
    "C:/Users/Kang/Desktop/DCIM/temp/_DSC2809.JPG",
    "C:/Users/Kang/Desktop/DCIM/temp/_DSC2844.JPG",
    "C:/Users/Kang/Desktop/DCIM/temp/_DSC2845.JPG",
    "C:/Users/Kang/Desktop/DCIM/temp/_DSC2848.JPG",
    "C:/Users/Kang/Desktop/DCIM/temp/_DSC2851.JPG",
    "C:/Users/Kang/Desktop/DCIM/temp/_DSC2852.JPG",
    "C:/Users/Kang/Desktop/DCIM/temp/_DSC2857.JPG",
]

# 加载CLIP的image encoder
clip_model = CLIPModel.from_pretrained("./clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("./clip-vit-large-patch14")

images = processor(images=[Image.open(i) for i in url], return_tensors="pt")


with torch.no_grad():
    image_features = clip_model.get_image_features(**images)
    text_features = text_encoder(text).logits
    
    print("Image features before normalization:", image_features)
    print("Text features before normalization:", text_features)

    # 归一化
    image_norms = image_features.norm(dim=1, keepdim=True)
    text_norms = text_features.norm(dim=1, keepdim=True)
    
    print("Image norms:", image_norms)
    print("Text norms:", text_norms)

    image_features = image_features / image_norms
    text_features = text_features / text_norms

    print("Image features after normalization:", image_features)
    print("Text features after normalization:", text_features)

    # 计算余弦相似度 logit_scale是尺度系数
    logit_scale = clip_model.logit_scale.exp()
    print("Logit scale:", logit_scale)

    # logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_image = logit_scale * text_features @ image_features.t()
    print("Logits per image before softmax:", logits_per_image)

    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    print("Probabilities:", np.around(probs, 3))