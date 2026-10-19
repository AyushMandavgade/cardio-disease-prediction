# Cardiovascular Disease Prediction (ML Lab Mini Project)

Predicts whether a patient has cardiovascular disease from 11 simple health features
(age, gender, height, weight, blood pressure, cholesterol, glucose, smoking, alcohol, activity).
Dataset: Kaggle "Cardiovascular Disease dataset" (70,000 rows).

## Project structure
```
data/cardio_train.csv     raw data (semicolon separated)
preprocess.py             cleaning + feature engineering + train/val/test split
train_models.py           trains & compares models, makes plots, saves best model
app/app.py                Streamlit web app (UI)
models/best_model.joblib  saved best model (created by train_models.py)
outputs/                  metrics tables and plots used in the report
```

## How to run
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python preprocess.py
python train_models.py
streamlit run app/app.py
```

## Pipeline
1. **Cleaning** - removed duplicates, impossible blood pressures (e.g. negative, 16020) and
   unrealistic height/weight; converted age from days to years; added BMI.
2. **Splitting** - stratified 70% train / 15% validation / 15% test (random_state=42).
3. **Models** - Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting,
   HistGradientBoosting, MLP (+ XGBoost and LightGBM if installed).
4. **Selection** - best model chosen on the validation ROC-AUC; final numbers reported on the untouched test set.
5. **Deployment** - Streamlit app (responsive, works on phone and desktop browsers).

## Results (test set)
See `outputs/model_comparison.csv`. All models reach about 73-74% accuracy and ~0.80 ROC-AUC.

## Deployment (free, cross-platform)
1. Push this folder to GitHub.
2. Go to https://share.streamlit.io, sign in with GitHub, click "New app".
3. Select the repo, set main file path to `app/app.py`, deploy.
4. Share the public URL - it opens on any phone, tablet or laptop.

## Disclaimer
Educational project, not a medical tool.
