import joblib
import numpy as np
import os

DEPLOYMENT_DIR = "deployment"

print("\n" + "="*70)
print("ALIGNED BASELINE - SYNCHRONIZED WITH FLASK API")
print("="*70)

# 1. Load model artifacts
try:
    model = joblib.load(os.path.join(DEPLOYMENT_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(DEPLOYMENT_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(DEPLOYMENT_DIR, "feature_names.pkl"))
    class_labels_file = joblib.load(os.path.join(DEPLOYMENT_DIR, "class_labels.pkl"))
    print("✓ Artifacts loaded successfully.")
except Exception as e:
    print(f"❌ Error loading artifacts: {e}")
    exit()

# 2. FIXED MASTER MAPPING (Identical to Flask API)
mapping = {
    "male": 1, "female": 0,
    "typical angina": 1, "atypical angina": 2, "non-anginal": 3, "asymptomatic": 4,
    "true": 1, "false": 0, "yes": 1, "no": 0,
    "normal": 0, "st-t wave": 1, "hypertrophy": 2,
    "upsloping": 1, "flat": 2, "downsloping": 3,
    "fixed defect": 3, "normal defect": 6, "reversable defect": 7
}

# 3. TEST DATA (Mixed Types - Case B Profile)
test_data = {
    "age": 63,
    "sex": "male",
    "cp": "typical angina",
    "trestbps": 145.0,
    "chol": 233,
    "fbs": "true",
    "restecg": "hypertrophy",
    "thalach": 150,
    "exang": "no",
    "oldpeak": 2.3,
    "slope": "downsloping",
    "ca": 0.0,
    "thal": "fixed defect"
}

# 4. DYNAMIC FEATURE ALIGNMENT (Identical logic to Flask API loop)
input_values = []
for f in feature_names:
    val = test_data.get(f)
    
    if val is None:
        print(f"⚠️ Missing feature: {f}. Using 0.0")
        input_values.append(0.0)
        continue
    
    # Map strings to numbers; leave numbers as floats (API logic)
    if isinstance(val, str):
        low_val = val.lower().strip()
        if low_val in mapping:
            converted = float(mapping[low_val])
        else:
            try:
                converted = float(val)
            except:
                converted = 0.0
    else:
        converted = float(val)
    
    input_values.append(converted)

# 5. PREDICTION PIPELINE
input_array = np.array(input_values).reshape(1, -1)
input_scaled = scaler.transform(input_array)

probabilities = model.predict_proba(input_scaled)[0]
predicted_idx = np.argmax(probabilities)

# 6. RESOLVE LABELS (API logic for internal model classes)
if hasattr(model, 'classes_'):
    actual_classes = [str(c).lower().strip() for c in model.classes_]
else:
    actual_classes = [str(c).lower().strip() for c in (class_labels_file if isinstance(class_labels_file, list) else list(class_labels_file.values()))]

# 7. OUTPUT RESULTS
print("-" * 70)
print(f"{'Class Name':<20} | {'Probability':<12}")
print("-" * 70)
for i, prob in enumerate(probabilities):
    label = actual_classes[i]
    marker = " ⭐ [PREDICTED]" if i == predicted_idx else ""
    print(f"{label:<20} | {prob*100:>10.2f}%{marker}")

print("="*70)
final_result = actual_classes[predicted_idx].upper()
print(f"FINAL RESULT: {final_result}")
print(f"CONFIDENCE: {probabilities[predicted_idx]*100:.2f}%")
print("="*70)