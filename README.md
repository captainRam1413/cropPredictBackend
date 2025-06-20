# 🌾 [Crop & Fertilizer Prediction Backend](https://crop-predict-frontend.vercel.app/)

Welcome to the [**Crop & Fertilizer Prediction Backend**](https://crop-predict-frontend.vercel.app/) — an intelligent agricultural recommendation system powered by machine learning and Django REST API. 
[Demo](https://crop-predict-frontend.vercel.app/)  
[**Frontend repo**](https://github.com/captainRam1413/cropPredictFrontend)

---

## 🚀 Features

- ✅ **Predicts the best crop and fertilizer** based on soil and weather conditions
- 🌍 Supports districts in **Western Maharashtra**
- 🔁 **Label encoding** for categorical data (District & Soil Color)
- 📈 **Trained ML model** using Scikit-learn’s MultiOutputClassifier
- 📡 **Django REST API** for seamless integration with web/mobile frontends

---

## 📦 Dataset Overview

The dataset contains samples from various districts in Western Maharashtra:

| Feature         | Description                                  |
|-----------------|----------------------------------------------|
| `District Name` | Name of the district                         |
| `Soil Color`    | Soil color present in that district          |
| `Nitrogen`      | Nitrogen content in the soil (mg/kg)         |
| `Potassium`     | Potassium level (mg/kg)                      |
| `Phosphorus`    | Phosphorus level (mg/kg)                     |
| `pH`            | pH value of the soil                         |
| `Rainfall`      | Average rainfall in mm                       |
| `Temperature`   | Average temperature in °C                    |
| `Crop`          | Recommended crop label (target)              |
| `Fertilizer`    | Recommended fertilizer (target)              |
| `Link`          | Educational YouTube link (not used in model) |

---

## 🤖 Machine Learning Model

### 🎯 Objective

Predict a tuple of **(Crop, Fertilizer)** using:
- Categorical features → LabelEncoded
- Numerical features → Used directly

### 📐 Input Vector

`[District (encoded), Soil Color (encoded), Nitrogen, Potassium, Phosphorus, pH, Rainfall, Temperature]`

### 🔧 Model Architecture

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

model = MultiOutputClassifier(RandomForestClassifier())
```

- **Label Encoders:**  
  - `encoder1.pkl` → District  
  - `encoder2.pkl` → Soil Color  
  - `map_crops.pkl` → Crop label mapping  
  - `map_fertilizers.pkl` → Fertilizer label mapping

#### 🔍 Prediction Flow

```python
input = ['Kolhapur', 'Light Brown', 30, 10, 5, 6.0, 600, 25]
input[0] = encoder1.transform([input[0]])
input[1] = encoder2.transform([input[1]])
input = np.array(input).reshape(1, -1)
prediction = model.predict(input)[0]
```

---

## 🌐 API Usage

### 📍 Endpoint

```
POST http://127.0.0.1:8000/api/predict/
```

**Headers:**
```
Content-Type: application/json
```

**Sample Request:**
```json
{
  "district": "Kolhapur",
  "soil": "Light Brown",
  "nitrogen": 30,
  "potassium": 10,
  "phosphorus": 5,
  "ph": 6.0,
  "rainfall": 600,
  "temperature": 25
}
```

**Sample Response:**
```json
{
  "predicted_crop": "Moong",
  "predicted_fertilizer": "MOP"
}
```

---

## 🛠️ Project Structure

```
.
├── agri_backend/            # Django project
│   └── settings.py
├── predictor/               # Django app
│   ├── models/              # Saved ML models
│   │   ├── cropmodel.pkl    
│   │   ├── fertilizermodel.pkl
│   │   ├── encoder1.pkl
│   │   ├── encoder2.pkl
│   │   ├── map_crops.pkl
│   │   └── map_fertilizers.pkl
│   ├── views.py             # API view logic
│   ├── ml_logic.py          # Model prediction logic
│   ├── urls.py              # App routing
├── manage.py
└── requirements.txt
```

---

## 💾 Installation & Setup

1. **Clone the Repo**
    ```bash
    git clone https://github.com/captainRam1413/cropPredictBackend.git
    cd cropPredictBackend
    ```

2. **Create Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run Server**
    ```bash
    python manage.py runserver
    ```

---

## 🧪 Model Files

If you train the model yourself, save these:

```python
joblib.dump(model_tuned, 'cropmodel.pkl')
joblib.dump(model_tuned, 'fertilizermodel.pkl')
joblib.dump(encoder1, 'encoder1.pkl')
joblib.dump(encoder2, 'encoder2.pkl')
joblib.dump(map_crops, 'map_crops.pkl')
joblib.dump(map_fertilizers, 'map_fertilizers.pkl')
```

---

## 📊 Model Performance

### 🌾 Crop Prediction Accuracy

| Model                   | Accuracy   |
|-------------------------|-----------|
| Decision Tree           | 99.38%    |
| Random Forest           | 99.73%    |
| Gradient Boosting       | 99.91% ✅ (Best) |
| Logistic Regression     | 76.62%    |
| SVM                     | 62.18%    |
| K-Nearest Neighbors     | 95.22%    |

### 💊 Fertilizer Prediction Accuracy

| Model                   | Accuracy   |
|-------------------------|-----------|
| Decision Tree           | 94.42% ✅ (Best) |
| Random Forest           | 91.94%    |
| Gradient Boosting       | 74.93%    |
| Logistic Regression     | 33.39%    |
| SVM                     | 30.12%    |
| K-Nearest Neighbors     | 52.61%    |

**🏆 Selected Best Models**
| Task                  | Best Model         | Accuracy |
|-----------------------|-------------------|----------|
| Crop Prediction       | Gradient Boosting | 99.91%   |
| Fertilizer Prediction | Decision Tree     | 94.42%   |

---

## 📌 Future Improvements

- 🌍 Integrate real-time weather API
- 📱 Build a mobile frontend for farmers
- 🧠 Explore deep learning models for improved accuracy
- 🔐 Add user authentication and role management

---

## 💡 Why This Matters

This system empowers farmers with AI-driven decisions that:
- Reduce crop failure due to poor selection
- Improve soil and fertilizer management
- Provide district-specific recommendations
- Assist rural areas with limited access to expert agronomists

---

**Made with ❤️ for smart agriculture.**
