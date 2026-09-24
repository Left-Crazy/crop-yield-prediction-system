# 🌾 Crop Yield Prediction System

An end-to-end Hybrid Machine Learning web application designed to predict agricultural crop yields (Tons / Hectare) using soil properties, climate variables, and fertilizer inputs.

---

## 🔗 Live Web Application
Evaluators can test the interactive system directly in any web browser without installing Python:
👉 **[Launch Live Web Demo](<https://crop-yield-predictionsystem-gdz37679bmz5jhtakkvzqc.streamlit.app/>)**

---

## 📌 Project Features
* **Hybrid Architecture:** Combines Random Forest Regression with Agronomic Guardrails to handle biological toxicity and environmental limits.
* **Interactive UI:** Dynamic Streamlit web application with real-time feedback and failure warnings.
* **Data Preprocessing:** Column Transformers with `OneHotEncoder` for handling categorical variables (State, District, Crop, Season, Soil Type).
* **Model Persistence:** Pipeline serialisation using `joblib`.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn (Random Forest Regressor, Pipeline, ColumnTransformer)
* **Data Processing:** Pandas, NumPy
* **Model Export:** Joblib
* **Web UI / Deployment:** Streamlit, Streamlit Community Cloud

---

## 💻 How to Run Locally

If you prefer to execute and test the code on your local environment:

### 1. Clone the Repository
```bash
git clone <https://github.com/Left-Crazy/Crop-Yield-Prediction_System.git>
cd "Crop Yeaild Prediction System"