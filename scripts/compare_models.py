import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import sys
sys.path.append(r"d:\Major Project sem 6 spit - Copy\Smart_Agri_Assistant")
from src.disease_keras import load_disease_model

OLD_MODEL_PATH = r"d:\Major Project sem 6 spit - Copy\Smart_Agri_Assistant\models\disease\leaf_disease_classifier.keras"
NEW_MODEL_PATH = r"d:\Major Project sem 6 spit - Copy\Smart_Agri_Assistant\models\final_multitask_crop_disease.h5"
METADATA_PATH = r"d:\Major Project sem 6 spit - Copy\Smart_Agri_Assistant\models\disease\leaf_disease_metadata.json"

TEST_IMAGES = [
    r"d:\Major Project sem 6 spit - Copy\Smart_Agri_Assistant\data\external\plantvillage dataset\color\Tomato___Early_blight\001187a0-57ab-4329-baff-e7246a9edeb0___RS_Erly.B 8178.JPG",
    r"d:\Major Project sem 6 spit - Copy\Smart_Agri_Assistant\data\external\plantvillage dataset\color\Potato___Late_blight\0051e5e8-d1c4-4a84-bf3a-a426cd332d73___RS_LB 4640.JPG"
]

def load_old_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["class_metadata"]

old_metadata = load_old_metadata()

# Load models
print("Loading old model...")
old_model = load_disease_model(OLD_MODEL_PATH, compile=False)

print("Loading new model...")
new_model = load_model(NEW_MODEL_PATH, compile=False)

def preprocess_old(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def preprocess_new(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    return tf.expand_dims(img, axis=0)

# The new model crops/diseases lists depend on how they were sorted alphabetically during training
unique_crops = ['Apple', 'Banana', 'Blueberry', 'Cherry (including sour)', 'Corn (maize)', 'Cotton', 'Grape', 'Orange', 'Peach', 'Pepper, bell', 'Potato', 'Raspberry', 'Rice', 'Soybean', 'Squash', 'Strawberry', 'Tomato']
unique_diseases = ['Apple scab', 'Bacterial Leaf Blight', 'Bacterial spot', 'Banana Black Sigatoka Disease', 'Banana Bract Mosaic Virus Disease', 'Banana Healthy Leaf', 'Banana Insect Pest Disease', 'Banana Moko Disease', 'Banana Panama Disease', 'Banana Yellow Sigatoka Disease', 'Black rot', 'Brown Spot', 'Cercospora leaf spot Gray leaf spot', 'Common rust', 'Early blight', 'Esca (Black Measles)', 'Healthy Rice Leaf', 'Late blight', 'Leaf Blast', 'Leaf blight (Isariopsis Leaf Spot)', 'Leaf scald', 'Leaf smut', 'Northern Leaf Blight', 'Powdery mildew', 'Sheath Blight', 'Spider mites Two-spotted spider mite', 'Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'aphids', 'bacterial leaf blight', 'brown spot', 'diseased', 'fresh', 'healthy', 'leaf smut', 'red rot', 'sigatoka', 'target spot', 'unknown']

for path in TEST_IMAGES:
    if not os.path.exists(path):
        print(f"Skipping {path}, file not found.")
        continue
        
    print(f"\n--- Testing: {os.path.basename(path)} ---")
    
    # Old Model Prediction
    old_input = preprocess_old(path)
    old_preds = old_model.predict(old_input, verbose=0)[0]
    top_idx = np.argmax(old_preds)
    old_meta = old_metadata[top_idx]
    print(f"OLD MODEL: {old_meta['crop_name']} - {old_meta['disease_name']} ({old_preds[top_idx]:.4f})")
    
    # New Model Prediction
    new_input = preprocess_new(path)
    new_preds = new_model.predict(new_input, verbose=0)
    crop_probs, disease_probs = new_preds[0][0], new_preds[1][0]
    
    # Since we didn't save the exact mapping in train script, we will just print the max index.
    c_idx = np.argmax(crop_probs)
    d_idx = np.argmax(disease_probs)
    
    # Trying to map it
    c_name = unique_crops[c_idx] if c_idx < len(unique_crops) else str(c_idx)
    d_name = unique_diseases[d_idx] if d_idx < len(unique_diseases) else str(d_idx)
    
    print(f"NEW MODEL: {c_name} - {d_name} (Crop Conf: {crop_probs[c_idx]:.4f}, Disease Conf: {disease_probs[d_idx]:.4f})")

