import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

print("TensorFlow Version:", tf.__version__)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4

BASE_DIR = r"d:\Major Project sem 6 spit"
PLANT_VILLAGE = r"d:\Major Project sem 6 spit - Copy\Smart_Agri_Assistant\data\external\plantvillage dataset\color"
BANANA = os.path.join(BASE_DIR, "Banana Disease Recognition Dataset", "Original Images", "Original Images")
COTTON = os.path.join(BASE_DIR, "Cotton Disease", "train")
RICE_AUG = os.path.join(BASE_DIR, "Rice_Leaf_AUG")
RICE_DIS = os.path.join(BASE_DIR, "rice_leaf_diseases")

file_paths = []
crop_labels = []
disease_labels = []

def add_images(folder_path, crop, disease):
    if not os.path.isdir(folder_path): return
    for img in os.listdir(folder_path):
        if img.lower().endswith(('.jpg', '.png', '.jpeg')):
            file_paths.append(os.path.join(folder_path, img))
            crop_labels.append(crop)
            disease_labels.append(disease)

# 1. PlantVillage
if os.path.exists(PLANT_VILLAGE):
    for folder in os.listdir(PLANT_VILLAGE):
        if "___" in folder:
            crop, disease = folder.split("___", 1)
        else:
            crop = folder
            disease = "unknown"
        add_images(os.path.join(PLANT_VILLAGE, folder), crop.replace("_", " ").strip(), disease.replace("_", " ").strip())

# 2. Banana
if os.path.exists(BANANA):
    for folder in os.listdir(BANANA):
        add_images(os.path.join(BANANA, folder), "Banana", folder.replace("Banana ", "").strip())

# 3. Cotton
if os.path.exists(COTTON):
    for folder in os.listdir(COTTON):
        add_images(os.path.join(COTTON, folder), "Cotton", folder.replace(" cotton leaf", "").replace(" cotton plant", "").strip())

# 4. Rice AUG
if os.path.exists(RICE_AUG):
    for folder in os.listdir(RICE_AUG):
        add_images(os.path.join(RICE_AUG, folder), "Rice", folder.replace("Rice Leaf", "").strip())

# 5. Rice Diseases
if os.path.exists(RICE_DIS):
    for folder in os.listdir(RICE_DIS):
        add_images(os.path.join(RICE_DIS, folder), "Rice", folder.strip())

unique_crops = sorted(list(set(crop_labels)))
unique_diseases = sorted(list(set(disease_labels)))

crop2idx = {name: idx for idx, name in enumerate(unique_crops)}
disease2idx = {name: idx for idx, name in enumerate(unique_diseases)}

NUM_CROPS = len(unique_crops)
NUM_DISEASES = len(unique_diseases)

print(f"Total Images Collected: {len(file_paths)}")
print(f"Total Crops: {NUM_CROPS}")
print(f"Total Diseases: {NUM_DISEASES}")

if len(file_paths) == 0:
    print("No images found. Exiting.")
    exit()

y_crop = np.array([crop2idx[l] for l in crop_labels])
y_disease = np.array([disease2idx[l] for l in disease_labels])

X_train, X_val, y_crop_train, y_crop_val, y_dis_train, y_dis_val = train_test_split(
    file_paths, y_crop, y_disease, test_size=0.2, stratify=y_crop, random_state=42
)

def parse_image(file_path, crop_label, disease_label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return img, (crop_label, disease_label)

def create_tf_dataset(paths, crop_labels, disease_labels, batch_size=BATCH_SIZE, is_training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, crop_labels, disease_labels))
    if is_training: ds = ds.shuffle(10000)
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = create_tf_dataset(X_train, y_crop_train, y_dis_train, BATCH_SIZE, is_training=True)
val_ds = create_tf_dataset(X_val, y_crop_val, y_dis_val, BATCH_SIZE, is_training=False)

def build_multi_task_model(num_crops, num_diseases):
    inputs = layers.Input(shape=IMG_SIZE + (3,))
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ])
    x = data_augmentation(inputs)
    
    backbone = applications.EfficientNetB0(include_top=False, weights='imagenet', input_tensor=x)
    backbone.trainable = False 
    
    x = layers.GlobalAveragePooling2D()(backbone.output)
    shared = layers.Dense(512, activation='relu')(x)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(0.3)(shared)
    
    crop_head = layers.Dense(256, activation='relu')(shared)
    crop_head = layers.Dropout(0.2)(crop_head)
    crop_output = layers.Dense(num_crops, activation='softmax', name='crop_output')(crop_head)
    
    disease_shared = layers.Dense(256, activation='relu')(shared)
    concat = layers.Concatenate()([disease_shared, crop_output])
    
    disease_head = layers.Dense(256, activation='relu')(concat)
    disease_head = layers.Dropout(0.2)(disease_head)
    disease_output = layers.Dense(num_diseases, activation='softmax', name='disease_output')(disease_head)
    
    return models.Model(inputs=inputs, outputs=[crop_output, disease_output])

model = build_multi_task_model(NUM_CROPS, NUM_DISEASES)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss={'crop_output': 'sparse_categorical_crossentropy', 'disease_output': 'sparse_categorical_crossentropy'},
    loss_weights={'crop_output': 1.0, 'disease_output': 1.5},
    metrics={'crop_output': 'accuracy', 'disease_output': 'accuracy'}
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
]

print("--- Training Phase 1: Frozen Backbone ---")
model.fit(train_ds, validation_data=val_ds, epochs=1, callbacks=callbacks)

# Ensure models dir exists
os.makedirs(r'd:\Major Project sem 6 spit - Copy\Smart_Agri_Assistant\models', exist_ok=True)
model.save(r'd:\Major Project sem 6 spit - Copy\Smart_Agri_Assistant\models\final_multitask_crop_disease.h5')
print("Model successfully trained and saved locally!")
