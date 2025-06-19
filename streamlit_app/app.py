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
#API_URL = "http://localhost:8000"

st.title("Projet 7 Openclassroom : Implémenter un modèle de score")

def show_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probabilité de défaillance du client (en %)", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickvals': [0, 26, 51, 76, 100], 
            'ticktext': ["0", "26", "51", "76", "100"]},
            'bar': {'color': "black",  'thickness': 0},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 26], 'color': "green"},    #client à risque très faible
                {'range': [26, 51], 'color': "yellow"},  #client à risque modéré
                {'range': [51, 76], 'color': "orange"},  #client à risque élevé
                {'range': [76, 100], 'color': "red"},    #client à risque très elévé
            ],
            'threshold': {
                'line': {'color': "white", 'width': 8},
                'thickness': 0.9,
                'value': probability * 100
            }
            
        }
    ))
    # Ajouter des labels personnalisés sous la jauge
    fig.update_layout(height=250, font = {'color': "white", 'family': "Arial"}, 
        annotations=[
            dict(x=0.265, y=0.35, text="Faible", showarrow=False, font=dict(size=12, color="black")),
            dict(x=0.42, y=0.85, text="Modéré", showarrow=False, font=dict(size=12, color="black")),
            dict(x=0.58, y=0.85, text="Élevé", showarrow=False, font=dict(size=12, color="black")),
            dict(x=0.745, y=0.35, text="Extrême", showarrow=False, font=dict(size=12, color="black")),
        ],
        margin={'t':50, 'b':15}
    )

    st.plotly_chart(fig)

def CreateProgressBar(pg_caption, pg_int_percentage, pg_colour, pg_bgcolour):
    pg_int_percentage = str(pg_int_percentage).zfill(2)
    pg_html = f"""<table style="width:100%; border-style: none;">
                        <tr style='font-weight:bold;'>
                            <td style='background-color:{pg_bgcolour};'>{pg_caption}: <span style='accent-color: {pg_colour}; bgcolor: transparent;'>
                                <progress value='{pg_int_percentage}' max='100'>{pg_int_percentage}%</progress> </span>{pg_int_percentage}% 
                            </td>
                        </tr>
                    </table><br>"""
    return pg_html


#Sélecteur d’ID client
client_ids = load_client_ids()
selected_id = st.selectbox("Choisissez un identifiant client :", client_ids)

#Bouton de prédiction
if st.button("Valider"):
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
            #st.markdown(CreateProgressBar("Probabilité de défaillance ", proba*100, "A5D6A7", ""), True)

            #st.success(f"Décision : {decision}")
            if proba <= 0.51:
                icone_response = "&#x2705;"
                color_reponse = "yellow"
                if proba <= 0.26:
                    color_reponse = "green"
            if proba > 0.51:
                icone_response = "&#x274C;"
                color_reponse = "red"
                if proba <= 0.76:
                    color_reponse = "orange"
            
            pg_html=f"""<div style="text-align: center"> <font size="6">Décision : <b><font color="{color_reponse}">{decision}</font></b> {icone_response}</font></div>"""
            st.markdown(pg_html, unsafe_allow_html=True)
        
        else:
            st.error(f"Erreur API : {response.status_code}")
            st.json(response.json())

    except Exception as e:
        st.error(f"Erreur de connexion à l'API : {e}")