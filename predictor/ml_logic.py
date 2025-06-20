import numpy as np
import joblib
import os
import google.generativeai as genai


from google import genai
from google.generativeai import types

client = genai.Client(api_key="AIzaSyALQTiFY8OX5wuoBER6bTkC90NnV77tv4s")
# Load files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# model = joblib.load(os.path.join(BASE_DIR, 'models/model_tuned.pkl'))
encoder1 = joblib.load(os.path.join(BASE_DIR, 'models/encoder1.pkl'))  # District
encoder2 = joblib.load(os.path.join(BASE_DIR, 'models/encoder2.pkl'))  # Soil
map_crops = joblib.load(os.path.join(BASE_DIR, 'models/map_crops.pkl'))
map_fert = joblib.load(os.path.join(BASE_DIR, 'models/map_fertilizers.pkl'))
CROP_MODEL = joblib.load('predictor/models/cropmodel.pkl')
FERTILIZER_MODEL = joblib.load('predictor/models/fertilizermodel.pkl')


# genai.configure(api_key='AIzaSyALQTiFY8OX5wuoBER6bTkC90NnV77tv4s')


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
        input_data = {
            'district': data['district'],
            'soil': data['soil'],
            'nitrogen': data['nitrogen'],
            'potassium': data['potassium'],
            'phosphorus': data['phosphorus'],
            'ph': data['ph'],
            'rainfall': data['rainfall'],
            'temperature': data['temperature']
        }
        predicted_data = {
            'predicted_crop': crop,
            'predicted_fertilizer': fert
        }
        advice = get_gemini_advice(input_data, predicted_data)
        print(f"Predicted crop: {crop}, Predicted fertilizer: {fert}")

        return {
            "predicted_crop": crop,
            "predicted_fertilizer": fert,
            "advice": advice
        }

    except Exception as e:
        return {"error": str(e)}
    
    
def get_gemini_advice(input_data, predicted_data):
    """
    input_data: dict with keys:
    'district', 'soil', 'nitrogen', 'potassium', 'phosphorus', 'ph', 'rainfall', 'temperature'
    
    predicted_data: dict with keys:
    'predicted_crop', 'predicted_fertilizer'
    """
    
    try:
        prompt = f"""
        You are an expert agricultural advisor assisting a farmer from {input_data['district']} district. 

        Using the following soil and environmental data:
        - Soil Type: {input_data['soil']}
        - Nitrogen: {input_data['nitrogen']} mg/kg
        - Potassium: {input_data['potassium']} mg/kg
        - Phosphorus: {input_data['phosphorus']} mg/kg
        - pH Level: {input_data['ph']}
        - Rainfall: {input_data['rainfall']} mm
        - Temperature: {input_data['temperature']} °C

        The machine learning model recommends:
        - Crop: **{predicted_data['predicted_crop']}**
        - Fertilizer: **{predicted_data['predicted_fertilizer']}**

        As an expert, provide a short, practical advisory (3–4 sentences) that includes:
        1. Specific guidance for preparing the soil and using the recommended fertilizer.
        2. Tips for successfully cultivating the predicted crop in these conditions.
        3. If the prediction seems suboptimal based on soil chemistry or weather, suggest a more suitable crop/fertilizer or additional amendments (e.g., composting, micronutrients, irrigation scheduling).
        4. Mention any district-specific practices or precautions relevant to the region if applicable.

        Ensure the advice is clear, farmer-friendly, and focused on improving productivity and crop health.
        Format the response as a single paragraph without bullet points or lists, using simple language suitable for farmers.
        Do not include any disclaimers or general information about agriculture.
        provide short, practical advice based on the provided data and predictions.
        """
        
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.text
    
    except Exception as e:
        return {"error": str(e)}
