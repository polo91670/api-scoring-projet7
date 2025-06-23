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
        #number={'font': {'size': 50}},  # Taille de police réduite ici
        domain={'x': [0, 1], 'y': [0, 1]},
        #title={'text': "Probabilité de défaillance du client (en %)", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickvals': [0, 9, 100], 
            'ticktext': ["0", "9", "100"], 'tickfont': {'size': 18}},
            'bar': {'color': "white",  'thickness': 0.7},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 9], 'color': "#49C289"},    #client à risque faible=décision accepté
                #{'range': [26, 51], 'color': "yellow"},  #client à risque modéré
                #{'range': [51, 76], 'color': "orange"},  #client à risque élevé
                {'range': [9, 100], 'color': "#D83E69"},    #client à risque elévé=décision refusé
            ],
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 0.9,
                'value': probability * 100
            }
            
        }
    ))
    # Ajouter des labels personnalisés sous la jauge
    #fig.update_layout(height=250, font = {'color': "white", 'family': "Arial"}, 
    #    annotations=[
      #      dict(x=0.24, y=1.3, text="Probabilité de défaillance du client (en %)", showarrow=False, font=dict(size=20, color="white"))
     #       dict(x=0.246, y=0.01, text="Faible", showarrow=False, font=dict(size=12, color="black")),
    #        dict(x=0.42, y=0.85, text="Modéré", showarrow=False, font=dict(size=12, color="black")),
    #        dict(x=0.58, y=0.85, text="Élevé", showarrow=False, font=dict(size=12, color="black")),
    #        dict(x=0.748, y=0.01, text="Élevé", showarrow=False, font=dict(size=12, color="black")),
    #    ],
    #    margin={'t':50, 'b':15} 
    #)
    # Pas de titre dans la figure elle-même
    fig.update_layout(height=200, margin=dict(t=20, b=20, l=20, r=20)
                     )

    # Ajoute une annotation pour la valeur à la position voulue
    #fig.add_annotation(x=0.5, y=0.5, text=round(probability*100,1), showarrow=False,
    #               font=dict(size=40, color="white"))

    #fig.add_annotation(
     #   x=0.5, y=0,
     #   text="🟩 Risque faible &nbsp;&nbsp;🟥 Risque élevé",
     #   showarrow=False,
     #   font=dict(size=13),
     #   xref="paper", yref="paper"
    #)

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

# CSS pour ajuster la hauteur et la marge de la selectbox
st.markdown("""
    <style>
    /* Largeur et centrage de la selectbox */
    div[data-baseweb="select"] {
        width: 110px !important;
        margin: 0;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    /* Aligner verticalement le contenu de la selectbox */
    div[data-baseweb="select"] > div {
        display: flex;
        align-items: center;
        height: 36px;  /* ajuster la hauteur au besoin */
    }
    </style>
""", unsafe_allow_html=True)


# Centrage via colonnes
left, center, right = st.columns([1, 2, 1])

with center:
    # Liste déroulante (selectbox)
    col1, col2 = st.columns([0.8, 2.2]) 
    with col1:
        #st.markdown("Choisissez un identifiant client :")
         # Utilise un conteneur HTML avec alignement vertical centré
        st.markdown("""
            <div style='display: flex; align-items: center; height: 36px;'>
                <p style='margin: 45px; font-weight: bold; white-space: nowrap;'>Choisissez un identifiant client :</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        selected_id = st.selectbox("", client_ids)

st.markdown("""
    <style>
    div.stButton > button {
        display: block;
        margin-left: 300px;
        margin-right: auto;
    }
    </style>
""", unsafe_allow_html=True)

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

            #st.title("Probabilité de défaillance du client (en %)")
            st.markdown(f"<h1 style='text-align: center; margin-top: 0px; margin-bottom: 0px; font-size: 20px;'>Probabilité de défaillance du client { selected_id} (en %)</h1>", unsafe_allow_html=True)
            
            show_gauge(proba)  #Affiche la jauge ici

            #st.success(f"Décision : {decision}")
            if proba <= 0.09:
                icone_response = "&#x2705;"
                color_reponse = "#49C289"
            if proba > 0.09:
                icone_response = "&#x274C;"
                color_reponse = "#D83E69"
            
            pg_html=f"""<div style="text-align: center"> <font size="6">Décision : <b><font color="{color_reponse}">{decision}</font></b> {icone_response}</font></div>"""
            st.markdown(pg_html, unsafe_allow_html=True)

            # Légende manuelle
            st.markdown("""
            <div style='text-align: left; margin-top: 10px; font-size: 15px;'>
            <b><u>Légende</u> :</b><br>
            🟩 [0 - 9%] : Risque faible de défaillance<br>
            🟥 ]9 - 100%] : Risque élevé de défaillance 
            </div>
            """, unsafe_allow_html=True)
    
        else:
            st.error(f"Erreur API : {response.status_code}")
            st.json(response.json())

    except Exception as e:
        st.error(f"Erreur de connexion à l'API : {e}")