import numpy as np
import joblib
import os

# Load files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# model = joblib.load(os.path.join(BASE_DIR, 'models/model_tuned.pkl'))
encoder1 = joblib.load(os.path.join(BASE_DIR, 'models/encoder1.pkl'))  # District
encoder2 = joblib.load(os.path.join(BASE_DIR, 'models/encoder2.pkl'))  # Soil
map_crops = joblib.load(os.path.join(BASE_DIR, 'models/map_crops.pkl'))
map_fert = joblib.load(os.path.join(BASE_DIR, 'models/map_fertilizers.pkl'))
CROP_MODEL = joblib.load('predictor/models/cropmodel.pkl')
FERTILIZER_MODEL = joblib.load('predictor/models/fertilizermodel.pkl')


def predict_crop_fertilizer(data):
    """
    data: dict with keys:
    'district', 'soil', 'nitrogen', 'potassium', 'phosphorus', 'ph', 'rainfall', 'temperature'
    """

    try:
        # Encode inputs
        dist = encoder1.transform([data['district']])[0]
        soil = encoder2.transform([data['soil']])[0]

        features = [
            dist,
            soil,
            float(data['nitrogen']),
            float(data['potassium']),
            float(data['phosphorus']),
            float(data['ph']),
            float(data['rainfall']),
            float(data['temperature'])
        ]

        features = np.array(features).reshape(1, -1)
        pred = CROP_MODEL.predict(features)[0]
        fert_pred = FERTILIZER_MODEL.predict(features)[0]

        # Decode crop
        crop = next(c[0] for c in map_crops if int(c[1]) == int(pred[0]))
        fert = next(f[0] for f in map_fert if int(f[1]) == int(fert_pred[1]))
        print(f"Predicted crop: {crop}, Predicted fertilizer: {fert}")

        return {
            "predicted_crop": crop,
            "predicted_fertilizer": fert
        }

    except Exception as e:
        return {"error": str(e)}
