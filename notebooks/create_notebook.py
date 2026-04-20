import nbformat as nbf

nb = nbf.v4.new_notebook()

# Helper to create markdown cells
def md(text):
    return nbf.v4.new_markdown_cell(text)

# Helper to create code cells
def code(text):
    return nbf.v4.new_code_cell(text)

cells = []

# Title
cells.append(md("# Smart Agri Assistant: AI-Based Crop Recommendation System\n*For India (Maharashtra Focus)*\n\nThis notebook demonstrates the End-to-End Machine Learning pipeline including Data Engineering, Feature Engineering, EDA, Model Building, Evaluation, and Advanced Rule-based Logic."))

# Phase 0: Setup
cells.append(md("## PHASE 0: Setup & Imports"))
cells.append(code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filter_warnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib"""))

# Phase 1: Data Engineering
cells.append(md("## PHASE 1: Data Engineering\n\n1. Load datasets\n2. Standardize column names\n3. Synthesize missing generic features\n4. Merge Datasets\n5. Handle missing values & remove outliers"))

cells.append(code("""# 1. Load both datasets
path1 = r'D:\\Major Project sem 6 spit\\Crop and fertilizer dataset.csv'
path2 = r'D:\\Major Project sem 6 spit\\Crop Recommendation using Soil Properties and Weather Prediction (1).csv'

df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)

print(f"Dataset 1 shape: {df1.shape}")
print(f"Dataset 2 shape: {df2.shape}")"""))

cells.append(code("""# 2. Standardize column names: N, P, K, temperature, humidity, pH, rainfall, label
# For Dataset 1:
df1_std = pd.DataFrame()
df1_std['N'] = df1['Nitrogen']
df1_std['P'] = df1['Phosphorus']
df1_std['K'] = df1['Potassium']
df1_std['temperature'] = df1['Temperature']
df1_std['pH'] = df1['pH']
df1_std['rainfall'] = df1['Rainfall']
df1_std['label'] = df1['Crop'].str.strip().str.lower()
df1_std['Soil_color'] = df1['Soil_color'].str.strip().str.lower()
df1_std['District_Name'] = df1['District_Name'].str.strip().str.lower()

# Dataset 1 lacks humidity, so let's generate realistic humidity based on rainfall/temp
np.random.seed(42)
df1_std['humidity'] = 40 + (df1_std['rainfall'] / 50) - (df1_std['temperature'] / 2) + np.random.normal(0, 5, len(df1_std))
df1_std['humidity'] = df1_std['humidity'].clip(20, 100)

# For Dataset 2:
df2_std = pd.DataFrame()
df2_std['N'] = df2['N']
df2_std['P'] = df2['P']
df2_std['K'] = df2['K']
df2_std['pH'] = df2['Ph']
df2_std['label'] = df2['label'].str.strip().str.lower()
df2_std['Soil_color'] = df2['Soilcolor'].astype(str).str.strip().str.lower()

# Dataset 2 weather data processing
temp_cols = [c for c in df2.columns if 'T2M_MAX' in c or 'T2M_MIN' in c]
rain_cols = [c for c in df2.columns if 'PRECTOTCORR' in c]
hum_cols = [c for c in df2.columns if 'QV2M' in c]

if len(temp_cols) > 0:
    df2_std['temperature'] = df2[temp_cols].mean(axis=1)
if len(rain_cols) > 0:
    # Convert mm/day to total seasonal/yearly approx
    df2_std['rainfall'] = df2[rain_cols].sum(axis=1) * 30 
if len(hum_cols) > 0:
     # QV2M is specific humidity roughly proportional to relative humidity
    df2_std['humidity'] = df2[hum_cols].mean(axis=1) * 10
    df2_std['humidity'] = df2_std['humidity'].clip(20, 100)

df2_std['District_Name'] = 'unknown' # Will impute later
"""))

cells.append(code("""# 3. Merge datasets
df_merged = pd.concat([df1_std, df2_std], ignore_index=True)

# Remove generic or undefined labels
valid_crops = df_merged['label'].value_counts()
valid_crops = valid_crops[valid_crops > 10].index.tolist()
df_merged = df_merged[df_merged['label'].isin(valid_crops)]

print(f"Merged dataset shape: {df_merged.shape}")
print(f"Unique crops available: {df_merged['label'].nunique()} (e.g. { df_merged['label'].unique()[:5] })")"""))

cells.append(code("""# 4. Handle Missing Values
num_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'pH', 'rainfall']
cat_cols = ['label', 'Soil_color', 'District_Name']

# Median imputation for numerical features
for col in num_cols:
    df_merged[col].fillna(df_merged[col].median(), inplace=True)

# Mode imputation for categorical features
for col in cat_cols:
    df_merged[col].fillna(df_merged[col].mode()[0], inplace=True)"""))

cells.append(code("""# 5. Remove Duplicates & Outliers
print(f"Before duplicates dropping: {df_merged.shape}")
df_merged.drop_duplicates(inplace=True)
print(f"After duplicates dropping: {df_merged.shape}")

# Z-score outlier removal on numerical features (threshold = 3)
from scipy import stats
outlier_mask = (np.abs(stats.zscore(df_merged[num_cols])) < 3).all(axis=1)
df_clean = df_merged[outlier_mask].copy()
print(f"After outlier removal: {df_clean.shape}")"""))

# Phase 2: Feature Engineering
cells.append(md("## PHASE 2: Feature Engineering\n\nAdd new features (Soil Type, Season, Region) and encode components."))
cells.append(code("""# 1. Add Soil Type (Black, Red, Alluvial)
def map_soil(color):
    if pd.isna(color):
        return 'Alluvial'
    c = str(color).lower()
    if 'black' in c or 'dark' in c:
        return 'Black'
    elif 'red' in c:
        return 'Red'
    else:
        return 'Alluvial'

df_clean['Soil_Type'] = df_clean['Soil_color'].apply(map_soil)

# 2. Add Region based on District logic (Maharashtra focus)
vidarbha = ['akola', 'amravati', 'bhandara', 'buldhana', 'chandrapur', 'gadchiroli', 'gondia', 'nagpur', 'wardha', 'washim', 'yavatmal']
konkan = ['mumbai', 'palghar', 'raigad', 'ratnagiri', 'sindhudurg', 'thane']
western_mah = ['kolhapur', 'pune', 'sangli', 'satara', 'solapur', 'ahmednagar']

def map_region(dist):
    d = dist.lower()
    if d in vidarbha: return 'Vidarbha'
    if d in konkan: return 'Konkan'
    if d in western_mah: return 'Western Maharashtra'
    return 'Other'

df_clean['Region'] = df_clean['District_Name'].apply(map_region)

# Because many are 'unknown' from Dataset 2, let's randomly impute 'Other' to typical MH regions based on crop logic to enrich data
np.random.seed(42)
other_mask = df_clean['Region'] == 'Other'
if other_mask.sum() > 0:
    df_clean.loc[other_mask, 'Region'] = np.random.choice(['Vidarbha', 'Konkan', 'Western Maharashtra', 'Marathwada'], size=other_mask.sum())

# 3. Season Allocation (Kharif, Rabi, Zaid)
def get_season(row):
    # Simplistic heuristic mapping just for feature richness
    rain = row['rainfall']
    temp = row['temperature']
    if rain > 200: return 'Kharif' # Monsoon
    elif temp < 25: return 'Rabi'  # Winter
    else: return 'Zaid'            # Summer

df_clean['Season'] = df_clean.apply(get_season, axis=1)

df_clean.head()"""))

cells.append(code("""# 4. Encode categorical features
label_encoder = LabelEncoder()

# Target Label
df_clean['Crop_Encoded'] = label_encoder.fit_transform(df_clean['label'])

# One-hot encode Categoricals
categorical_features = ['Soil_Type', 'Season', 'Region']
df_encoded = pd.get_dummies(df_clean, columns=categorical_features, drop_first=True)

# Drop redundant textual columns
X = df_encoded.drop(['label', 'Crop_Encoded', 'Soil_color', 'District_Name'], axis=1)
y = df_clean['Crop_Encoded']

print(f"Features ready for modeling: {X.shape}")"""))

# Phase 3: EDA
cells.append(md("## PHASE 3: Exploratory Data Analysis (EDA)"))
cells.append(code("""# 1. Crop Distribution
plt.figure(figsize=(12, 6))
sns.countplot(data=df_clean, y='label', order=df_clean['label'].value_counts().index[:20])
plt.title('Top 20 Crop Distributions')
plt.xlabel('Count')
plt.ylabel('Crop')
plt.show()"""))

cells.append(code("""# 2. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_matrix = df_clean[num_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Agricultural Features')
plt.show()"""))

cells.append(code("""# 3. Regional Patterns (Pie chart of Regions)
plt.figure(figsize=(8,8))
df_clean['Region'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('pastel'))
plt.title('Distribution of Agricultural Regions')
plt.ylabel('')
plt.show()"""))

# Phase 4: Model Building
cells.append(md("## PHASE 4: Model Building\n\nTraining Random Forest, XGBoost, SVM, and KNN."))
cells.append(code("""# 1. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Normalization/Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)"""))

cells.append(code("""# Initialize Models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Train and Evaluate
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {
        'Model': model,
        'Accuracy': acc,
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1-score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    print(f"{name} completed. Accuracy: {acc*100:.2f}%")"""))

# Phase 5: Model Evaluation
cells.append(md("## PHASE 5: Model Evaluation"))
cells.append(code("""# Compare Model Performances
results_df = pd.DataFrame(results).T[['Accuracy', 'Precision', 'Recall', 'F1-score']]
display(results_df.sort_values(by='Accuracy', ascending=False))

# Identify best model
best_model_name = results_df['Accuracy'].idxmax()
best_model = results[best_model_name]['Model']
print(f"\\n🏆 Best Model Selected: {best_model_name} with Accuracy {results_df.loc[best_model_name, 'Accuracy']*100:.2f}%")"""))

cells.append(code("""# Confusion Matrix of Best Model
y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()"""))

cells.append(code("""# Feature Importance (If RF or XGBoost)
if best_model_name in ['Random Forest', 'XGBoost']:
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.show()"""))

# Phase 6: Advanced Logic
cells.append(md("## PHASE 6: Advanced Logic\nCombining Rule-Based Engine + ML Model Predictions"))
cells.append(code("""def advanced_predict_crop(n, p, k, temp, hum, ph, rain, soil_type, season, region):
    '''
    Combines rule-based heuristics with ML output.
    Returns Top 3 recommended crops.
    '''
    # 1. Setup Input DataFrame matching training features
    input_data = {col: [0] for col in X.columns}
    input_data['N'] = [n]
    input_data['P'] = [p]
    input_data['K'] = [k]
    input_data['temperature'] = [temp]
    input_data['humidity'] = [hum]
    input_data['pH'] = [ph]
    input_data['rainfall'] = [rain]
    
    soil_col = f'Soil_Type_{soil_type}'
    if soil_col in input_data: input_data[soil_col] = [1]
        
    season_col = f'Season_{season}'
    if season_col in input_data: input_data[season_col] = [1]
        
    region_col = f'Region_{region}'
    if region_col in input_data: input_data[region_col] = [1]
        
    input_df = pd.DataFrame(input_data)
    input_scaled = scaler.transform(input_df)
    
    # 2. Get Probabilities from Best ML model
    probs = best_model.predict_proba(input_scaled)[0]
    
    # Top 3 indices
    top_3_indices = np.argsort(probs)[-3:][::-1]
    top_3_crops = label_encoder.inverse_transform(top_3_indices)
    top_3_probs = probs[top_3_indices]
    
    predictions = {crop: prob for crop, prob in zip(top_3_crops, top_3_probs)}
    
    # 3. Rule-Based Amplification (Maharashtra Focus)
    rule_applied = False
    
    # Rule 1: High rain, High Humidity -> Rice boost
    if rain > 150 and hum > 80:
        if 'rice' in predictions: predictions['rice'] += 0.20
        else: predictions['rice'] = 0.20
        rule_applied = True
        
    # Rule 2: Vidarbha + Black Soil -> Cotton boost
    if region == 'Vidarbha' and soil_type == 'Black':
        if 'cotton' in predictions: predictions['cotton'] += 0.25
        else: predictions['cotton'] = 0.25
        rule_applied = True
    
    # Sort again after rules
    sorted_preds = sorted(predictions.items(), key=lambda item: item[1], reverse=True)[:3]
    
    result = [crop for crop, prob in sorted_preds]
    return result, "Rules triggered!" if rule_applied else "ML Only"

# Test Advanced Function
test_n, test_p, test_k, test_t, test_h, test_ph, test_r = 90, 42, 43, 20.8, 82.0, 6.5, 202.9
recs, logic = advanced_predict_crop(test_n, test_p, test_k, test_t, test_h, test_ph, test_r, 'Black', 'Kharif', 'Konkan')
print(f"Given Conditions -> Rain: 202mm, Hum: 82%")
print(f"Logic Used: {logic}")
print(f"Recommendations: {recs}")"""))

# Phase 7: Saving output
cells.append(md("## PHASE 7: Deployment Artifacts\nSave models and preprocessors to drive the UI."))
cells.append(code("""# Save the best model, scaler, label_encoder, and feature list
artifact_dir = r'd:\\Major Project sem 6 spit\\Smart_Agri_Assistant\\models\\'
joblib.dump(best_model, artifact_dir + 'best_model.pkl')
joblib.dump(scaler, artifact_dir + 'scaler.pkl')
joblib.dump(label_encoder, artifact_dir + 'label_encoder.pkl')
joblib.dump(list(X.columns), artifact_dir + 'feature_columns.pkl')

print("Artifacts successfully saved to models directory!")"""))

nb.cells.extend(cells)

# Save Notebook
with open('D:\\Major Project sem 6 spit\\Smart_Agri_Assistant\\notebooks\\01_end_to_end_pipeline.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook generated successfully!")
