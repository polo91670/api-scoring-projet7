from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.data import load_data
from app.model import load_model, predict_score

app = FastAPI(
    title="API Scoring Crédit",
    description="Projet 7 - Implémenter un modèle de score (OpenClassrooms) V1.1",
    version="1.1"
)

#Ajout du middleware CORS pour autoriser les appels depuis l’extérieur
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise tous les domaines (à restreindre en prod si besoin)
    allow_credentials=True,
    allow_methods=["*"],  # Autorise toutes les méthodes (POST, GET, etc.)
    allow_headers=["*"],  # Autorise tous les headers
)

# Chargement des ressources globales
df = load_data()
model = load_model()
available_ids = df["SK_ID_CURR"].tolist()

# Modèle de requête
class ClientID(BaseModel):
    client_id: int

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de scoring crédit"}

@app.get("/client_ids")
def get_client_ids():
    """Renvoie la liste des identifiants client disponibles."""
    return {"client_ids": available_ids}

@app.post("/predict_score")
def get_score(data: ClientID):
    """Prédit le score et renvoie la décision binaire selon le seuil."""
    client_id = data.client_id
    if client_id not in df["SK_ID_CURR"].values:
        raise HTTPException(status_code=404, detail="Client ID non trouvé.")
    
    # Extraire les features du client
    client_data = df[df["SK_ID_CURR"] == client_id].drop(columns=["SK_ID_CURR"]).values[0]
    
    # Prédiction
    result = predict_score(model, client_data)
    
    return {
        "client_id": client_id,
        "score_proba": result["score_proba"],
        "seuil_utilisé": result["seuil"],
        "décision": result["decision"]
    }