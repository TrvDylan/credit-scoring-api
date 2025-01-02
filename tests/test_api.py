from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from io import StringIO, BytesIO
from main import app
import pandas as pd


client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World!"}

def test_score_valid_file():
    # Données simulées pour le test
    data = pd.DataFrame({
        "SK_ID_CURR": [100001],
        "PAYMENT_RATE": [0.025],
        "EXT_SOURCE_1": [0.5],
        "EXT_SOURCE_2": [0.4],
        "EXT_SOURCE_3": [0.7],
        "DAYS_BIRTH": [-12000],
        "AMT_ANNUITY": [25000],
        "DAYS_EMPLOYED": [-4000],
        "DAYS_ID_PUBLISH": [-2000],
        "APPROVED_CNT_PAYMENT_MEAN": [10],
        "INSTAL_DAYS_ENTRY_PAYMENT_MAX": [-5],
        "ACTIVE_DAYS_CREDIT_MAX": [-300],
        "DAYS_EMPLOYED_PERC": [0.33],
        "ACTIVE_DAYS_CREDIT_ENDDATE_MIN": [-200],
        "INSTAL_DPD_MEAN": [0.5],
        "DAYS_REGISTRATION": [-4000],
        "ANNUITY_INCOME_PERC": [0.2],
        "REGION_POPULATION_RELATIVE": [0.01],
        "AMT_CREDIT": [500000],
        "CLOSED_DAYS_CREDIT_MAX": [-600],
        "PREV_CNT_PAYMENT_MEAN": [12]
    })
    
    # Simuler un fichier CSV en mémoire
    csv_file = BytesIO()
    data.to_csv(csv_file, index=False)
    csv_file.seek(0) 

    # Envoyer une requête POST avec le fichier CSV
    response = client.post("/score", files={"file": ("test.csv", csv_file, "text/csv")})
    
    # Vérifier la réponse
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, list)  # La réponse doit être une liste de dictionnaires
    assert "Client_ID" in result[0]
    assert "Classe_Predite" in result[0]
    assert "Probabilite_Classe_1" in result[0]

def test_score_missing_columns():
    # Données avec des colonnes manquantes
    data = pd.DataFrame({
        "SK_ID_CURR": [100001],
        "PAYMENT_RATE": [0.025]
        # D'autres colonnes obligatoires sont absentes
    })
    
    csv_file = BytesIO()
    data.to_csv(csv_file, index=False)
    csv_file.seek(0)

    response = client.post("/score", files={"file": ("test_invalid.csv", csv_file, "text/csv")})

    # Vérifier que l'API renvoie une erreur
    assert response.status_code == 400
    assert "Les colonnes suivantes sont manquantes" in response.json()["detail"]


def test_one_client():
    # Données simulées avec un résultat connu
    data = pd.DataFrame({
        "SK_ID_CURR": [334215],
        "PAYMENT_RATE": [0.048108],
        "EXT_SOURCE_1": [0.305633],
        "EXT_SOURCE_2": [0.155032],
        "EXT_SOURCE_3": [0.483050],
        "DAYS_BIRTH": [-8191],
        "AMT_ANNUITY": [38848.5],
        "DAYS_EMPLOYED": [-400],
        "DAYS_ID_PUBLISH": [-868],
        "APPROVED_CNT_PAYMENT_MEAN": [18],
        "INSTAL_DAYS_ENTRY_PAYMENT_MAX": [-8],
        "ACTIVE_DAYS_CREDIT_MAX": [-206],
        "DAYS_EMPLOYED_PERC": [0.048834],
        "ACTIVE_DAYS_CREDIT_ENDDATE_MIN": [3447.0],
        "INSTAL_DPD_MEAN": [0.235294],
        "DAYS_REGISTRATION": [-631],
        "ANNUITY_INCOME_PERC": [0.34532],
        "REGION_POPULATION_RELATIVE": [0.028663],
        "AMT_CREDIT": [807534.0],
        "CLOSED_DAYS_CREDIT_MAX": [-673],
        "PREV_CNT_PAYMENT_MEAN": [18]
    })
    
    # Résultat attendu pour cette observation (par exemple, Classe_Predite = 1)
    expected_prediction = {
        "Client_ID": 334215,
        "Classe_Predite": 1,
        # "Probabilite_Classe_1": 0.8  # Exemple de probabilité attendue
    }
    
    # Convertir les données en fichier CSV
    csv_file = BytesIO()
    data.to_csv(csv_file, index=False)
    csv_file.seek(0)

    # Envoyer la requête POST
    response = client.post("/score", files={"file": ("test.csv", csv_file, "text/csv")})

    # Vérifier la réponse
    assert response.status_code == 200
    result = response.json()

    # Vérifier que la prédiction correspond à ce qui est attendu
    assert len(result) == 1  # Une seule observation prédite
    assert result[0]["Client_ID"] == expected_prediction["Client_ID"]
    assert result[0]["Classe_Predite"] == expected_prediction["Classe_Predite"]
    # assert abs(result[0]["Probabilite_Classe_1"] - expected_prediction["Probabilite_Classe_1"]) < 0.05  # Tolérance de 5%
