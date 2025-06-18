import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import numpy as np

#Charger les ID clients depuis un fichier local
@st.cache_data
def load_client_ids():
    df = pd.read_csv("data.csv")
    df.rename(columns = {'SK_ID_CURR': 'client_id'}, inplace=True)
    return df["client_id"].astype(str).tolist()


#URL de ton API FastAPI déployée sur Render
API_URL = "https://api-scoring-projet7.onrender.com"
#API_URL = "http://localhost:8000/predict_score"

st.title("Projet 7 Openclassroom : Implémenter un modèle de score")

def show_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probabilité de défaillance du client (en %)", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickvals': [0, 25, 50, 75, 100], 
            'ticktext': ["0", "25", "50", "75", "100"]},
            'bar': {'color': "black",  'thickness': 0},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 25], 'color': "green"},    #client à risque très faible
                {'range': [25, 50], 'color': "yellow"},  #client à risque modéré
                {'range': [50, 75], 'color': "orange"},  #client à risque élevé
                {'range': [75, 100], 'color': "red"},    #client à risque très elévé
            ],
            'threshold': {
                'line': {'color': "black", 'width': 8},
                'thickness': 0.85,
                'value': probability * 100
            }
            
        }
    ))
    # Ajouter des labels personnalisés sous la jauge
    fig.update_layout(height=500, width=500, font = {'color': "white", 'family': "Arial"}, 
        annotations=[
            dict(x=0.05, y=0.3, text="Faible", showarrow=False, font=dict(size=14, color="black")),
            dict(x=0.35, y=0.8, text="Modéré", showarrow=False, font=dict(size=14, color="black")),
            dict(x=0.7, y=0.8, text="Élevé", showarrow=False, font=dict(size=14, color="black")),
            dict(x=0.95, y=0.3, text="Extrême", showarrow=False, font=dict(size=14, color="black")),
        ],
        margin={'t':0, 'b':30}
    )

    st.plotly_chart(fig)
    
#Sélecteur d’ID client
client_ids = load_client_ids()
selected_id = st.selectbox("Choisissez un identifiant client :", client_ids)

#Bouton de prédiction
if st.button("Prédire le score"):
    try:
        response = requests.post(f"{API_URL}/predict_score", json={"client_id": selected_id})
        #st.write("Statut HTTP :", response.status_code)
        #st.write("Texte brut :", response.text)

        if response.status_code == 200:
            result = response.json()
            proba = result.get("score_proba")
            decision = result.get("décision")

            #st.success(f"Probabilité de défaut : {proba:.2%}")

            show_gauge(proba)  #Affiche la jauge ici
            st.success(f"Décision : {decision}")

        
        else:
            st.error(f"Erreur API : {response.status_code}")
            st.json(response.json())

    except Exception as e:
        st.error(f"Erreur de connexion à l'API : {e}")