from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import numpy as np
import io
import sys
import logging
import joblib

# Logs
logging.basicConfig(level=logging.INFO)
sys.stdout = sys.stderr

# Variables
model_path = "models/model.pkl"
scaler_path = "models/scaler.pkl"
features = ["SK_ID_CURR","PAYMENT_RATE","EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3","DAYS_BIRTH","AMT_ANNUITY","DAYS_EMPLOYED","DAYS_ID_PUBLISH",
            "APPROVED_CNT_PAYMENT_MEAN","INSTAL_DAYS_ENTRY_PAYMENT_MAX","ACTIVE_DAYS_CREDIT_MAX","DAYS_EMPLOYED_PERC","ACTIVE_DAYS_CREDIT_ENDDATE_MIN",
            "INSTAL_DPD_MEAN","DAYS_REGISTRATION","ANNUITY_INCOME_PERC","REGION_POPULATION_RELATIVE","AMT_CREDIT","CLOSED_DAYS_CREDIT_MAX","PREV_CNT_PAYMENT_MEAN"]

# Création de l'API
app = FastAPI()

# Chargement du modele de ML et du scaler avec joblib
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    print(f"Erreur lors du chargement des fichiers : {e}")
    model = None
    scaler = None

@app.get("/")
def read_root():
    return {"message": "Hello World!"}

@app.post('/score')
async def score(file: UploadFile = File(...)):
    # Lecture du fichier csv
    contents = await file.read()
    data = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    # Vérifier les features
    missing_features = [feature for feature in features if feature not in data.columns]
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Les colonnes suivantes sont manquantes :{missing_features}"
        )
    
    # Récupérer l'id du client
    id_client = data["SK_ID_CURR"]

    print(type(id_client))

    # Filtrer les colonnes
    df = data[features]
    X = df.drop("SK_ID_CURR", axis=1).values

    print(df.head())

    # Standardiser les données
    X_scaled = scaler.transform(X)

    def generate_predictions(model, X, client_ids, seuil=0.55):
        # Récupérer les probabilités des classes
        probas = model.predict_proba(X)
        prob_class_1 = probas[:, 1]  # Probabilité d'appartenance à la classe 1
    
        # Calculer les classes prédites avec le seuil personnalisé
        predictions = (prob_class_1 > seuil).astype(int)
    
        # Construire un DataFrame avec les résultats
        results = pd.DataFrame({
            'Client_ID': client_ids,
            'Classe_Predite': predictions,
            'Probabilite_Classe_1': prob_class_1
        })
    
        return results
    
    predictions = generate_predictions(model, X_scaled, id_client, seuil=0.55)
    print(type(predictions))
    print(predictions)

    # Résultats
    return predictions.to_dict(orient='records')