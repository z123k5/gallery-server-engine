import io
from PIL import Image
import requests
import clip
import torch
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from torchsummary import summary

import tensorflow as tf

import os
os.environ["HF_HOME"] = "./"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
proxy_dict = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}

# 这里是输入文本的，可以随意替换。
# <P>
query_texts = [
    '一幅精美的插画，展现一片风景如画的湖畔，湖水清澈如镜，倒映着天空中的霞光。湖边盛开着五彩斑斓的鲜花，微风轻拂，花瓣随风飘舞。远处连绵的青山在薄雾中若隐若现，增添了一种梦幻般的氛围。画面整体色调柔和，光影交错，给人一种宁静而唯美的感觉。']
# <OP>
# query_texts = [
#     '''一张破败荒凉的画面，一片枯黄的荒野上，杂草丛生，土地干裂，仿佛许久未曾见过雨水。天空阴沉沉的，被厚重的乌云笼罩，透不出一丝阳光。远处几栋年久失修的废弃建筑残破不堪，窗户破碎，墙壁斑驳脱落，透露出岁月的侵蚀。空气中弥漫着灰尘与寂静，没有任何生机，整个场景透出一股萧条和压抑的氛围，让人感到沉闷而无望。''']





# query_texts = [
#     '一幅富有寓意的绘画，画面中央是一棵历经风霜的古树，它的枝干虽然扭曲苍老，却依然生机勃勃地向天空伸展，象征着坚韧不拔的精神。在树下，一块被风雨侵蚀的顽石静静伫立，表面布满岁月的痕迹，却依然坚固不移，隐喻着坚定不屈的品格。远处，一条潺潺流淌的小溪映照着落日余晖，象征着岁月流转和生命的延续。整幅画面通过自然之物传达深刻的哲理，使人沉思。']

# query_texts = ['啤酒','晚宴','合影','旅游']  # 这里是输入文本的，可以随意替换。
# 加载Taiyi 中文 text encoder
text_tokenizer = BertTokenizer.from_pretrained(
    "./Taiyi-CLIP-Roberta-large-326M-Chinese")
text_encoder = BertForSequenceClassification.from_pretrained(
    "./Taiyi-CLIP-Roberta-large-326M-Chinese").eval()
text = text_tokenizer(query_texts, return_tensors='pt', padding=True)['input_ids']

# url = [
#     "C:/Users/Kang/Desktop/DCIM/temp/_DSC2035.JPG",
#     "C:/Users/Kang/Desktop/DCIM/temp/_DSC2056.JPG",
#     "C:/Users/Kang/Desktop/DCIM/temp/_DSC2807.JPG",
#     "C:/Users/Kang/Desktop/DCIM/temp/_DSC2808.JPG",
#     "C:/Users/Kang/Desktop/DCIM/temp/_DSC2809.JPG",
#     "C:/Users/Kang/Desktop/DCIM/temp/_DSC2844.JPG",
#     "C:/Users/Kang/Desktop/DCIM/temp/_DSC2845.JPG",
#     "C:/Users/Kang/Desktop/DCIM/temp/_DSC2848.JPG",
#     "C:/Users/Kang/Desktop/DCIM/temp/_DSC2851.JPG",
#     "C:/Users/Kang/Desktop/DCIM/temp/_DSC2852.JPG",
#     "C:/Users/Kang/Desktop/DCIM/temp/_DSC2857.JPG",
# ]

# 从文件夹中获取所有图片的路径
def get_foldimg_uri():
    foldimg_uri = []
    for root, dirs, files in os.walk("C:/Users/Kang/Desktop/DCIM/temp"):
        for file in files:
            if file.endswith(".JPG"):
                foldimg_uri.append(os.path.join(root, file))
    return foldimg_uri

url = get_foldimg_uri()

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
    print("Torch Probabilities:", np.around(probs, 3))


    # Calc using Tensorflow
    # TFlite model

    class SimilarityModel(tf.Module):
        def __init__(self):
            super().__init__()
            self.logit_scale = tf.Variable(logit_scale, dtype=tf.float32)

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[1, 768], dtype=tf.float32),
            tf.TensorSpec(shape=[None, 768], dtype=tf.float32)
        ])
        def compute_similarity(self, text_feat, image_feat):
            # text_feat = tf.nn.l2_normalize(text_feat, axis=1)  # 归一化
            # image_feat = tf.nn.l2_normalize(image_feat, axis=1)  # 归一化
            logits = logit_scale * (text_feat @ tf.transpose(image_feat))
            # return tf.nn.softmax(logits, axis=-1)
            return tf.nn.softmax(logits, axis=-1)
    
    # 转换为 TFLite
    similarity_model = SimilarityModel()
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [similarity_model.compute_similarity.get_concrete_function()]
    )
    # 启用动态形状支持
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.experimental_new_converter = True

    # 允许动态批次大小
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()

    # 运行 TFLite 计算
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 设置输入
    interpreter.resize_tensor_input(
        input_details[1]['index'], (11, 768))  # 例如 image_features 有 10 个
    interpreter.allocate_tensors()


    interpreter.set_tensor(
        input_details[0]['index'], text_features.cpu().numpy().astype(np.float32))
    interpreter.set_tensor(
        input_details[1]['index'], image_features.cpu().numpy().astype(np.float32))

    # 运行
    interpreter.invoke()

    # 获取输出
    probs = interpreter.get_tensor(output_details[0]['index'])
    print("TFLite Probabilities:", np.around(probs, 3))
