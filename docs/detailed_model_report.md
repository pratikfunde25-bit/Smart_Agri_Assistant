# Comprehensive Detailed Report: Model Performances

This report contains the evaluation metrics (Accuracy, Precision, Recall, F1-Score) and Annotated Confusion Matrices for all the machine learning models developed in the Smart Agri Assistant project.

## Crop Recommendation (XGBoost)

### Overall Performance
- **Accuracy**: 0.9459
- **Precision (Macro)**: 0.8011
- **Recall (Macro)**: 0.7573
- **F1 Score (Macro)**: 0.7717

### Exploratory Data Analysis & Feature Insights
The following visualizations highlight the data distribution and feature correlations that power the crop recommendation engine:

**Dataset Distribution**
![Crop Distribution](../visualizations/crop_distribution.png)

**Feature Correlation Heatmap**
![Correlation Heatmap](../visualizations/correlation_heatmap.png)

**Regional Patterns**
![Regional Patterns](../visualizations/regional_patterns.png)

### Annotated Confusion Matrix
The confusion matrix is annotated with exact counts of True Positives, True Negatives, False Positives, and False Negatives per class.

![Crop Recommendation (XGBoost) Confusion Matrix](crop_recommendation_cm_annotated.png)

### Class-wise Detailed Report
| Class Name | Precision | Recall | F1-Score | Support (TP + FN) |
| :--- | :--- | :--- | :--- | :--- |
| Barley | 0.9820 | 0.9454 | 0.9633 | 403 |
| Bean | 0.9840 | 0.9109 | 0.9460 | 202 |
| Cotton | 0.8500 | 0.8947 | 0.8718 | 19 |
| Dagussa | 0.9107 | 0.8947 | 0.9027 | 57 |
| Fallow | 1.0000 | 0.9048 | 0.9500 | 21 |
| Ginger | 0.6000 | 0.6000 | 0.6000 | 5 |
| Gram | 0.4286 | 0.6000 | 0.5000 | 5 |
| Grapes | 1.0000 | 1.0000 | 1.0000 | 5 |
| Groundnut | 0.7143 | 0.4167 | 0.5263 | 12 |
| Jowar | 0.5000 | 0.5294 | 0.5143 | 17 |
| Maize | 0.9120 | 0.9580 | 0.9344 | 595 |
| Moong | 0.5000 | 0.2000 | 0.2857 | 5 |
| Niger seed | 0.9792 | 0.9216 | 0.9495 | 51 |
| Pea | 1.0000 | 0.9600 | 0.9796 | 75 |
| Potato | 1.0000 | 0.9211 | 0.9589 | 38 |
| Red Pepper | 1.0000 | 0.7391 | 0.8500 | 23 |
| Rice | 0.6000 | 0.6667 | 0.6316 | 9 |
| Sorghum | 1.0000 | 0.8276 | 0.9057 | 58 |
| Sugarcane | 0.9545 | 0.8077 | 0.8750 | 26 |
| Teff | 0.9650 | 0.9841 | 0.9745 | 1008 |
| Tur | 0.7143 | 0.6250 | 0.6667 | 8 |
| Turmeric | 0.4000 | 0.5000 | 0.4444 | 4 |
| Urad | 0.2857 | 0.4000 | 0.3333 | 5 |
| Wheat | 0.9463 | 0.9667 | 0.9564 | 601 |

---

## Hybrid/Joint Disease & Crop Model (DenseNet121)

### Overall Performance
- **Accuracy**: 0.7719
- **Precision (Macro)**: 0.7737
- **Recall (Macro)**: 0.7895
- **F1 Score (Macro)**: 0.7510

- **Auxiliary Crop Head Accuracy**: 0.8947

### Training Performance & Explainability (Grad-CAM)
The following curve visualizes the accuracy and loss metrics throughout the training epochs, demonstrating the model's convergence and stability due to the joint-task regularization:

**Training Curves**
![Training Curves](../reports/disease_joint/training_curves.png)

**Explainability Example (Grad-CAM Heatmap)**
To ensure our model predictions are transparent and visually verifiable by farmers, we use Grad-CAM to highlight the regions of the leaf that triggered the disease classification.
![Grad-CAM Explainability Heatmap](../visualizations/sample_gradcam.jpg)

### Annotated Confusion Matrix
The confusion matrix is annotated with exact counts of True Positives, True Negatives, False Positives, and False Negatives per class.

![Hybrid/Joint Disease & Crop Model (DenseNet121) Confusion Matrix](joint_disease_cm_annotated.png)

### Class-wise Detailed Report
| Class Name | Precision | Recall | F1-Score | Support (TP + FN) |
| :--- | :--- | :--- | :--- | :--- |
| Apple___Apple_scab | 1.0000 | 1.0000 | 1.0000 | 1 |
| Apple___Black_rot | 0.4000 | 1.0000 | 0.5714 | 2 |
| Apple___Cedar_apple_rust | 1.0000 | 0.5000 | 0.6667 | 2 |
| Apple___healthy | 1.0000 | 0.5000 | 0.6667 | 2 |
| Blueberry___healthy | 1.0000 | 1.0000 | 1.0000 | 1 |
| Cherry_(including_sour)___Powdery_mildew | 0.0000 | 0.0000 | 0.0000 | 2 |
| Cherry_(including_sour)___healthy | 1.0000 | 1.0000 | 1.0000 | 1 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 0.6667 | 1.0000 | 0.8000 | 2 |
| Corn_(maize)___Common_rust_ | 0.5000 | 1.0000 | 0.6667 | 1 |
| Corn_(maize)___Northern_Leaf_Blight | 0.0000 | 0.0000 | 0.0000 | 1 |
| Corn_(maize)___healthy | 1.0000 | 1.0000 | 1.0000 | 2 |
| Grape___Black_rot | 1.0000 | 0.5000 | 0.6667 | 2 |
| Grape___Esca_(Black_Measles) | 0.5000 | 1.0000 | 0.6667 | 1 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 1.0000 | 1.0000 | 1.0000 | 1 |
| Grape___healthy | 1.0000 | 1.0000 | 1.0000 | 2 |
| Orange___Haunglongbing_(Citrus_greening) | 1.0000 | 1.0000 | 1.0000 | 1 |
| Peach___Bacterial_spot | 1.0000 | 1.0000 | 1.0000 | 1 |
| Peach___healthy | 1.0000 | 1.0000 | 1.0000 | 1 |
| Pepper,_bell___Bacterial_spot | 1.0000 | 1.0000 | 1.0000 | 2 |
| Pepper,_bell___healthy | 1.0000 | 1.0000 | 1.0000 | 1 |
| Potato___Early_blight | 0.5000 | 1.0000 | 0.6667 | 2 |
| Potato___Late_blight | 1.0000 | 0.5000 | 0.6667 | 2 |
| Potato___healthy | 1.0000 | 1.0000 | 1.0000 | 2 |
| Raspberry___healthy | 1.0000 | 1.0000 | 1.0000 | 2 |
| Soybean___healthy | 1.0000 | 1.0000 | 1.0000 | 1 |
| Squash___Powdery_mildew | 1.0000 | 1.0000 | 1.0000 | 1 |
| Strawberry___Leaf_scorch | 1.0000 | 1.0000 | 1.0000 | 1 |
| Strawberry___healthy | 1.0000 | 1.0000 | 1.0000 | 1 |
| Tomato___Bacterial_spot | 1.0000 | 0.5000 | 0.6667 | 2 |
| Tomato___Early_blight | 1.0000 | 1.0000 | 1.0000 | 2 |
| Tomato___Late_blight | 0.0000 | 0.0000 | 0.0000 | 1 |
| Tomato___Leaf_Mold | 1.0000 | 0.5000 | 0.6667 | 2 |
| Tomato___Septoria_leaf_spot | 0.3333 | 1.0000 | 0.5000 | 1 |
| Tomato___Spider_mites Two-spotted_spider_mite | 0.0000 | 0.0000 | 0.0000 | 1 |
| Tomato___Target_Spot | 0.5000 | 1.0000 | 0.6667 | 2 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 1.0000 | 1.0000 | 1.0000 | 2 |
| Tomato___Tomato_mosaic_virus | 1.0000 | 1.0000 | 1.0000 | 1 |
| Tomato___healthy | 0.0000 | 0.0000 | 0.0000 | 2 |

---

