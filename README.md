## API de Scoring Crédit – Projet 7 OpenClassrooms
Cette API permet de prédire la probabilité de remboursement d’un client à partir de ses caractéristiques. Elle a été développée dans le cadre du Projet 7 du parcours Data Scientist d’OpenClassrooms : **"Implémenter un modèle de score"**.

## Objectifs du projet
- Déployer un modèle de scoring dans une API.
- Utiliser un **seuil de probabilité optimal** pour automatiser la décision d'acceptation.
- Exposer les résultats via une interface simple (ex: Streamlit).

## Technologies utilisées
- Notebook Jupyter	6.3.0
- Python 3.12.4
- FastAPI 0.115.12
- scikit-learn 1.4.2
- pandas 2.2.2
- joblib 1.4.2
- pytest 8.3.4

## Structure du projet api
API/
├── app/
│ ├── main.py # Point d’entrée FastAPI
│ ├── model.py # Chargement modèle + prédiction avec seuil
│ ├── data.py # Chargement des données clients
├── data.csv # Jeu de données clients (id_client + toutes les features issues de l'éatpe features engineering hors target)
├── LGBMClm_model.pkl # Modèle ML entraîné (via joblib)
├── threshold.txt # Chargement du seuil optimal de prédiction
├── requirements.txt # Dépendances Python
├── tests/
│ └── test_api.py # Tests unitaires de l’API
├── README.md

## Lancer l’API en local
- Dans un prompt dos %> uvicorn app.main:app --reload
- Puis accédez à la documentation interactive : http://127.0.0.1:8000/docs

## Lancer les tests unitaires
- 4 tests sont effectués : test de l'existance de l'API, test de l'existance d'une liste d'identifiant client, test de prédiction d'un score sur un client existant dans le jeu de données, test de prédiction d'un score sur un client non existant dans le jeu de données
- Dans un prompt dos %> pytest tests/
  