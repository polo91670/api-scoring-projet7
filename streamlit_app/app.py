import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

#Charger les ID clients depuis un fichier local
@st.cache_data
def load_client_ids():
    df = pd.read_csv("data.csv")
    df.rename(columns = {'SK_ID_CURR': 'client_id'})
    return df["client_id"].astype(str).tolist()


#URL de ton API FastAPI déployée sur Render
API_URL = "https://api-scoring-projet7.onrender.com/predict_score"

st.title("Score Client - Projet 7")

def show_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probabilité de défaut (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 25], 'color': "green"},    #client à risque très faible
                {'range': [25, 50], 'color': "yellow"},  #client à risque modéré
                {'range': [50, 75], 'color': "orange"},  #client à risque élevé
                {'range': [75, 100], 'color': "red"},    #client à risque très elévé
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    st.plotly_chart(fig)
    
#Sélecteur d’ID client
client_ids = load_client_ids()
selected_id = st.selectbox("Choisissez un identifiant client :", client_ids)

#Bouton de prédiction
if st.button("Prédire"):
    try:
        response = requests.post(API_URL, json={"client_id": selected_id})
        
        if response.status_code == 200:
            result = response.json()
            proba = result.get("score_proba")
            decision = result.get("décision")

            st.success(f"Probabilité de défaut : {proba:.2%}")
            st.write("Décision :", decision)

            show_gauge(proba)  #Affiche la jauge ici

        else:
            st.error(f"Erreur API : {response.status_code}")
            st.json(response.json())

    except Exception as e:
        st.error(f"Erreur de connexion à l'API : {e}")