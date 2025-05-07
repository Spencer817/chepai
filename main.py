import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, callbacks

# 数据路径配置
original_dataset_dir = '../Downloads/dataset'
base_dir = '../Downloads/split_dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir   = os.path.join(base_dir, 'val')
test_dir  = os.path.join(base_dir, 'test')

# 数据划分比例
def split_dataset():
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(train_dir), os.makedirs(val_dir), os.makedirs(test_dir)

    for class_name in os.listdir(original_dataset_dir):
        class_path = os.path.join(original_dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        images = os.listdir(class_path)
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * 0.7)
        n_val   = int(n_total * 0.15)

        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train+n_val],
            'test': images[n_train+n_val:]
        }

        for split_name, split_images in splits.items():
            split_class_dir = os.path.join(base_dir, split_name, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            for img in split_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(split_class_dir, img)
                shutil.copyfile(src, dst)

split_dataset()

# 参数
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 10

# 数据增强
data_augment = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True
)

test_augment = ImageDataGenerator(rescale=1./255)

# 数据生成器
def get_generator(directory):
    return data_augment.flow_from_directory(
        directory,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

def get_test_generator(directory):
    return test_augment.flow_from_directory(
        directory,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

train_gen = get_generator(train_dir)
val_gen = get_generator(val_dir)
test_gen = get_test_generator(test_dir)

# 类别权重（解决轻微不平衡）
class_weight = {0: 1.0, 1: 1.0}  # 可根据比例调整

# 构建模型（ResNet50 + Attention）
def build_model():
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model

model = build_model()
model.compile(
    optimizer=optimizers.Adam(learning_rate=3e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 回调
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
]

# 第一阶段训练
model.fit(
    train_gen,
    epochs=EPOCHS_PHASE1,
    validation_data=val_gen,
    class_weight=class_weight,
    callbacks=callbacks_list
)

# 解冻底层微调
model.trainable = True
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(
    train_gen,
    epochs=EPOCHS_PHASE2,
    validation_data=val_gen,
    class_weight=class_weight,
    callbacks=callbacks_list
)

# 评估
loss, acc = model.evaluate(test_gen)
print(f"\n✅ Test Accuracy: {acc:.4f}")

# 保存模型
model.save('resnet_attention_plate_vs_street2.h5')
