#intégration de 4 tests unitaires avec pytest
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

#test N°1 que l'API existe
#un code réponse = 200 => test OK
#un code réponse = 404 => test KO
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

#test N°2 la liste des clients
def test_get_client_ids():
    response = client.get("/client_ids")
    assert response.status_code == 200
    data = response.json()
    assert "client_ids" in data
    assert isinstance(data["client_ids"], list)
    assert len(data["client_ids"]) > 0

#test N°3 un score de prédiction sur un client appartenant à une liste
def test_predict_score_valid():
    # Récupère un ID valide depuis /client_ids
    client_ids = client.get("/client_ids").json()["client_ids"]
    valid_id = client_ids[0]
    
    response = client.post("/predict_score", json={"client_id": valid_id})
    assert response.status_code == 200
    data = response.json()
    assert "client_id" in data
    assert "score_proba" in data
    assert "décision" in data
    assert isinstance(data["score_proba"], float)
    assert data["client_id"] == valid_id

#test N°4 un score de prédiction sur un client n'appartenant pas à une liste
def test_predict_score_invalid():
    response = client.post("/predict_score", json={"client_id": -1})
    assert response.status_code == 404
    assert response.json()["detail"] == "Client ID non trouvé."