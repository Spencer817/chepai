import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

IMG_SIZE = 224
model_path = 'resnet_attention_plate_vs_street2.h5'
img_dir = 'inference_images'  # 存放你想预测的图片
output_csv = 'prediction_results.csv'

# 加载模型
model = load_model(model_path)

# 预测函数
def preprocess_and_predict(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    prob = model.predict(x)[0][0]
    label = '车牌' if prob > 0.5 else '街景'
    return label, prob

# 推理整个文件夹
results = []
for fname in os.listdir(img_dir):
    if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
        path = os.path.join(img_dir, fname)
        label, prob = preprocess_and_predict(path)
        print(f"✅ {fname}: {label}（概率: {prob:.4f})")
        results.append((fname, label, prob))

# 保存结果为 CSV
with open(output_csv, 'w', encoding='utf-8') as f:
    f.write('filename,label,probability\n')
    for fname, label, prob in results:
        f.write(f'{fname},{label},{prob:.4f}\n')

print(f"\n📄 所有预测结果已保存至 {output_csv}")