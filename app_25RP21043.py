from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from datetime import datetime
import traceback
import os

app = Flask(__name__)
CORS(app)

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEPLOYMENT_DIR = os.path.join(BASE_DIR, "deployment")

# FIXED MASTER MAPPING: 
# Matches the strings from your HTML <select> tags to numerical values
mapping = {
    "male": 1, "female": 0,
    "typical angina": 1, "atypical angina": 2, "non-anginal": 3, "asymptomatic": 4,
    "true": 1, "false": 0, "yes": 1, "no": 0,
    "normal": 0, "st-t wave": 1, "hypertrophy": 2, 
    "upsloping": 1, "flat": 2, "downsloping": 3,
    "fixed defect": 3, "normal defect": 6, "reversable defect": 7
}

def load_model_artifacts():
    try:
        model = joblib.load(os.path.join(DEPLOYMENT_DIR, "best_model.pkl"))
        scaler = joblib.load(os.path.join(DEPLOYMENT_DIR, "scaler.pkl"))
        feature_names = joblib.load(os.path.join(DEPLOYMENT_DIR, "feature_names.pkl"))
        class_labels = joblib.load(os.path.join(DEPLOYMENT_DIR, "class_labels.pkl"))
        return model, scaler, feature_names, class_labels
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        return None, None, None, None

MODEL, SCALER, FEATURE_NAMES, CLASS_LABELS = load_model_artifacts()

if MODEL and hasattr(MODEL, 'classes_'):
    ACTUAL_CLASSES = [str(c).lower().strip() for c in MODEL.classes_]
else:
    ACTUAL_CLASSES = [str(c).lower().strip() for c in (CLASS_LABELS if isinstance(CLASS_LABELS, list) else [])]

def get_risk_level_info(predicted_class_name):
    name = predicted_class_name.lower().strip()
    risk_map = {
        "immediate danger": {"risk": "Critical", "clinical": "Emergency condition. Immediate medical care required."},
        "severe": {"risk": "High", "clinical": "Severe heart disease detected. Urgent consultation recommended."},
        "mild": {"risk": "Medium", "clinical": "Mild heart disease detected. Lifestyle changes advised."},
        "very mild": {"risk": "Low Risk", "clinical": "Very mild indicators. Regular checkups advised."},
        "no disease": {"risk": "Healthy", "clinical": "No signs of heart disease detected."}
    }
    return risk_map.get(name, {"risk": "Unknown", "clinical": "Consult a physician for interpretation."})

# --- NEW ROUTE: HEALTH CHECK ---
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat()
    })

# --- NEW ROUTE: MODEL INFO ---
@app.route("/api/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model_name": "Random Forest Classifier (Optimized)",
        "test_accuracy": 0.88 
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        if MODEL is None:
            return jsonify({"success": False, "error": "Model not loaded"}), 500

        data = request.get_json(silent=True)
        if not data:
            return jsonify({"success": False, "error": "No JSON data"}), 400

        input_values = []
        for f in FEATURE_NAMES:
            val = data.get(f)
            if val is None:
                return jsonify({"success": False, "error": f"Missing: {f}"}), 400
            
            if isinstance(val, str):
                low_val = val.lower().strip()
                if low_val in mapping:
                    converted = float(mapping[low_val])
                else:
                    try: converted = float(val)
                    except: return jsonify({"success": False, "error": f"Invalid string value: {val}"}), 400
            else:
                converted = float(val)
            
            input_values.append(converted)

        input_array = np.array(input_values).reshape(1, -1)
        input_scaled = SCALER.transform(input_array)
        
        probabilities = MODEL.predict_proba(input_scaled)[0]
        predicted_idx = np.argmax(probabilities)
        predicted_class_name = ACTUAL_CLASSES[predicted_idx]

        risk_info = get_risk_level_info(predicted_class_name)

        return jsonify({
            "success": True,
            "predicted_class": predicted_class_name,
            "confidence": round(float(probabilities[predicted_idx]) * 100, 2),
            "risk_level": risk_info["risk"],
            "clinical_interpretation": risk_info["clinical"],
            "timestamp": datetime.now().isoformat(),
            "per_class_probabilities": {
                ACTUAL_CLASSES[i]: round(float(probabilities[i]) * 100, 2)
                for i in range(len(probabilities))
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)