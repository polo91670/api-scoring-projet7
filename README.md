## API de Scoring Crédit – Projet 7 parcours Data Scientist de OpenClassrooms
Cette API permet de prédire la probabilité de défaillance de remboursement d’un client à partir de ses caractéristiques. Elle a été développée dans le cadre du Projet 7 du parcours Data Scientist de OpenClassrooms : **"Implémenter un modèle de scoring"**.

## Objectifs du projet
- Déployer un modèle de scoring dans une API.
- Utiliser un **seuil de probabilité optimal** pour automatiser la décision d'acceptation/de refus.
- Exposer les résultats via une interface simple

## Technologies utilisées
- notebook Jupyter	6.3.0
- python 3.12.4
- fastAPI 0.115.12
- scikit-learn 1.4.2
- pandas 2.2.2
- joblib 1.4.2
- pytest 8.3.4
- mlflow 2.22.0
- shap 0.46.0
- joblib 1.4.2
- streamlit 1.45.1
- evidently 0.7.8

## Structure du projet concernant l'API
API
|--APP
|-----main.py # Point d’entrée FastAPI
|-----model.py # Chargement modèle + prédiction avec seuil
|-----data.py # Chargement des données clients
|-----requirements.txt # librairies nécessaires pour l'API
|--STREAMLIT_APP
|-----app.py # Point d’entrée interface streamlit
|-----requirements.txt # librairies nécessaires pour streamlit
|--TESTS
|data.csv # Jeu de données clients (id_client + toutes les features issues de l'éatpe features engineering hors target)
|LGBMClm_model.pkl # Modèle ML entraîné (via joblib)
|threshold.txt # Chargement du seuil optimal de prédiction
|render.yaml #instruction pour execution de l'api et streamlit sur render.com
|README.md

## Lancer l’API en local
- Dans un prompt dos %> uvicorn app.main:app --reload
- Puis accédez à la documentation interactive : http://127.0.0.1:8000/docs

## Lancer les tests unitaires
- 4 tests sont effectués : test de l'existance de l'API, test de l'existance d'une liste d'identifiant client, test de prédiction d'un score sur un client existant dans le jeu de données, test de prédiction d'un score sur un client non existant dans le jeu de données
- Dans un prompt dos %> pytest tests/

## Lancer l’API en production
- https://streamlit-scoring-projet7.onrender.com/

## Auteur
Projet réalisé dans le cadre du parcours **Data Scientist - OpenClassrooms**

- Nom : *VA NYIA LU*
- Contact : *paul.va-nyia-lu@cetelem.fr*