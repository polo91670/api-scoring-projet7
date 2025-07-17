#librairies pour le projet 7
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import numpy as np
#librairies pour le projet 8
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff

#Charger les ID clients depuis un fichier local
@st.cache_data
def load_client_ids():
    df = pd.read_csv("data_v2.csv")
    df.rename(columns = {'SK_ID_CURR': 'client_id'}, inplace=True)
    df_client = pd.read_csv("data_v2.csv", index_col="SK_ID_CURR")
    df_client.index = df_client.index.astype(str)
    df_client_with_target = pd.read_csv("data_v3.csv", index_col="SK_ID_CURR")
    df_client_with_target.index = df_client_with_target.index.astype(str)
    #df_client.index.name = "client_id"
    return df["client_id"].astype(str).tolist(), df_client, df_client_with_target, df

#charger le modèle
def load_model():
    model = joblib.load("LGBMClm_model.pkl")
    return model

pipeline = load_model()

# Seuil utilisé par le modèle
threshold = 0.51

#URL de ton API FastAPI déployée sur Render
API_URL = "https://api-scoring-projet7.onrender.com"
#API_URL = "http://localhost:8000"

st.title("Projet 7 Openclassroom : Implémenter un modèle de score")


# Initialiser l'état de session pour éviter la réinitialisation
if "validated" not in st.session_state:
    st.session_state["validated"] = False
    
def show_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        #number={'font': {'size': 50}},  # Taille de police réduite ici
        domain={'x': [0, 1], 'y': [0, 1]},
        #title={'text': "Probabilité de défaillance du client (en %)", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickvals': [0, 51, 100], 
            'ticktext': ["0", "51", "100"], 'tickfont': {'size': 14}},
            'bar': {'color': "white",  'thickness': 0.7},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 51], 'color': "#008BFB"},    #client à risque faible=décision accepté #49C289
                #{'range': [26, 51], 'color': "yellow"},  #client à risque modéré
                #{'range': [51, 76], 'color': "orange"},  #client à risque élevé
                {'range': [51, 100], 'color': "#D83E69"},    #client à risque elévé=décision refusé
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
client_ids, df_client, df_client_with_target, df = load_client_ids()

# Vérification
#st.dataframe(df_client.head(5))

#st.markdown("""
#    <style>
#    div.stButton > button {
#        display: block;
#        margin-left: 300px;
#        margin-right: auto;
#    }
#    </style>
#""", unsafe_allow_html=True)

#st.markdown("""
#    <style>
#    div.stButton > button {
#        font-size: 20px;
#        padding: 0.5em 1em;
#        width: 100%;
#        white-space: nowrap;
#    }
#    </style>
#""", unsafe_allow_html=True)

# CSS pour ajuster la hauteur et la marge de la selectbox
#st.markdown("""
#    <style>
#    /* Largeur et centrage de la selectbox */
#    div[data-baseweb="select"] {
#        width: 110px !important;
#        margin: 0;
#        padding-top: 0 !important;
#        padding-bottom: 0 !important;
#    }
#    /* Aligner verticalement le contenu de la selectbox */
#    div[data-baseweb="select"] > div {
#        display: flex;
#        align-items: center;
#        height: 36px;  /* ajuster la hauteur au besoin */
#    }
#    </style>
#""", unsafe_allow_html=True)


# Centrage via colonnes
left, center, right = st.columns([1, 2, 1])

with center:
    # Liste déroulante (selectbox)
    #------------------------
    col1, col2 = st.columns([1, 2]) 
    with col1:
        #st.markdown("Choisissez un identifiant client :")
         # Utilise un conteneur HTML avec alignement vertical centré
        st.markdown("""
            <div style='display: flex; align-items: center; height: 36px;'>
                <p style='margin: 40px; font-weight: bold; white-space: nowrap;'>Choisissez un identifiant client :</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        selected_id = st.selectbox("", client_ids)
        client_data = df_client.loc[[selected_id]]
        client_data_with_target = df_client_with_target.loc[[selected_id]]

        # Liste initiale des colonnes numériques disponibles
        numerical_features = client_data_with_target.select_dtypes(include='number').columns.tolist()
        
        # Retirer des colonnes inutiles
        exclude_columns = ['TARGET', 'SK_ID_CURR']
        available_features = [col for col in numerical_features if col not in exclude_columns]

        
    #------------------------
    # Titre centré
    #st.markdown(
    #    "<h4 style='text-align: center;'>Choisissez un identifiant client</h4>",
    #    unsafe_allow_html=True
    #)

    # Centrer la selectbox avec un wrapper HTML et du CSS
   # st.markdown("""
   #     <style>
   ##     .select-wrapper {
    #        display: flex;
    #        justify-content: center;
    #        margin-top: 10px;
    #    }
    #    .select-wrapper > div {
    #        width: 1% !important;  /* Largeur du bouton selectbox */
    #    }
    #    </style>
     #   <div class="select-wrapper">
   # """, unsafe_allow_html=True)

   # selected_id = st.selectbox("", client_ids, key="client_id_select")

   # st.markdown("</div>", unsafe_allow_html=True)

    # Tu peux récupérer les infos du client ici
  #  client_data = df_client.loc[[selected_id]]

    
    # ✅ Ligne pour les boutons centrés
    b_left, b1, b_spacer, b2, b_right = st.columns([0.3, 1.6, 0.2, 1.6, 0.3])
    with b1:
        if st.button("✅ Valider"):
            st.session_state["validated"] = True
    with b2:
        if st.button("🔄 Reset"):
            st.session_state["validated"] = False
            st.rerun()

            
#Bouton de prédiction
if st.session_state["validated"]:
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
            if proba <= threshold:
                icone_response = "&#x2714;" #"&#x2705;"
                color_reponse = "#008BFB" #49C289
                prediction = 0
            if proba > threshold:
                icone_response = "&#x274C;"
                color_reponse = "#D83E69"
                prediction = 1
            pg_html=f"""<div style="text-align: center"> <font size="6">Décision : <b><font color="{color_reponse}">{decision}</font></b> {icone_response}</font></div>"""
            st.markdown(pg_html, unsafe_allow_html=True)

            # Légende manuelle
            st.markdown("""
            <div style='text-align: left; margin-top: 10px; font-size: 15px;'>
            <b><u>Légende</u> :</b><br>
            🟦 [0 - 51%] : Risque faible de défaillance<br>
            🟥 ]51 - 100%] : Risque élevé de défaillance 
            </div><br>
            """, unsafe_allow_html=True)

            # Initialisation de SHAP
            #explainer = shap.TreeExplainer(model)
            #shap_values = explainer.shap_values(df_client)


            # Affichage des données du client
            #st.subheader(f"📄 Données du client sélectionné : {selected_id}")
            #st.dataframe(client_data)
            scaler = pipeline.named_steps['scaler']
            model = pipeline.named_steps['model']
            
            # Appliquer le scaler (MinMaxScaler) sur les données client
            try:
                X_scaled = scaler.transform(client_data)
                X_scaled_df = pd.DataFrame(X_scaled, columns=client_data.columns, index=client_data.index)
            except Exception as e:
                st.error(f"Erreur lors de la transformation des données : {e}")
                st.stop()
            
            # --- 4. Prédiction de la probabilité
            try:
                y_proba = model.predict_proba(X_scaled_df)
                if y_proba.shape[1] == 2:
                    proba = y_proba[0][1]  # Classe 1 = probabilité de défaut
                else:
                    proba = y_proba[0]     # Cas très rare (multi-class)
                #st.metric(label="📊 Probabilité de défaut (score modèle)", value=f"{proba:.1%}")
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")
                st.stop()
    
            # --- 5. SHAP TreeExplainer
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_scaled_df)
            except Exception as e:
                st.error(f"Erreur SHAP : {e}")
                st.stop()
                
            #----------------------------------------------------------------------------------------------------------
            with st.expander("🔎 Features importance local (cliquez pour visualiser)", expanded=False):
                try:
                    #Obtenir les valeurs SHAP et la valeur de base
                    if isinstance(shap_values, list) and len(shap_values) == 2:
                        shap_vals = shap_values[1][0]
                        expected_val = explainer.expected_value[1]
                    else:
                        shap_vals = shap_values[0]
                        expected_val = explainer.expected_value
                
                    #Créer l'explication SHAP
                    explanation = shap.Explanation(
                        values=shap_vals,
                        base_values=expected_val,
                        data=X_scaled_df.iloc[0],
                        feature_names=X_scaled_df.columns
                    )
                
                    #Calculer f(x) et la probabilité
                    fx = float(expected_val + shap_vals.sum())
                    proba_fx = 1 / (1 + np.exp(-fx))
                
                    #Deux colonnes : waterfall et bar chart
                    col1, col2 = st.columns(2)
                
                    #Colonne 1 : SHAP waterfall plot
                    with col1:
                        st.markdown("<span style='font-size:20px'>Graphique en cascade SHAP</span>", unsafe_allow_html=True)

                        fig, ax = plt.subplots(figsize=(7, 5.5))
                        shap.plots.waterfall(explanation, max_display=10, show=False)
                        fig.tight_layout()
                        st.pyplot(fig)

                        st.markdown(f"""
                        <div style="font-size:14px; color:white;">
                            <p>Ce graphique montre comment chaque variable contribue à la prédiction finale du modèle pour ce client.</p>
                            <ul>
                                <li>La valeur moyenne (base du modèle) est <code>E[f(x)] = {expected_val:.3f}</code>.</li>
                                <li>Chaque barre indique comment une variable fait <b>monter (en rouge)</b> ou <b>diminuer (en bleu)</b> la valeur de ce client par rapport à cette base.</li>
                                <li>La somme des contributions aboutit à la valeur finale pour ce client <code>f(x) = {fx:.3f}</code>.</li>
                                <li>La probabilité d'être défaillant est donc <code>P = {proba_fx:.1%}</code></li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                
                    #Colonne 2 : Bar chart coloré top 10 SHAP
                    with col2:
                        st.markdown("<span style='font-size:20px'>Top 10 variables importantes</span>", unsafe_allow_html=True)
                
                        top_features = pd.Series(shap_vals, index=X_scaled_df.columns).sort_values(key=abs, ascending=False).head(10)
                        colors = ['#D83E69' if v > 0 else '#008BFB' for v in top_features.values]
                
                        fig2, ax2 = plt.subplots(figsize=(7, 5.5))
                        bars = ax2.barh(top_features.index[::-1], top_features.values[::-1], color=colors[::-1])

                        # Affichage des valeurs à droite (positif) ou gauche (négatif)
                        for bar in bars:
                            width = bar.get_width()
                            value_str = f"{width:.2f}"
                            xpos = width + 0.01 if width > 0 else width - 0.01
                            align = 'left' if width > 0 else 'right'
                            color_val = '#D83E69' if width > 0 else '#008BFB'
                            
                            ax2.text(
                                xpos,
                                bar.get_y() + bar.get_height() / 2,
                                value_str,
                                va='center',
                                ha=align,
                                fontsize=10,
                                color=color_val,
                                fontweight='bold'
                            )
                        
                        ax2.set_xlabel("Valeur SHAP")
                        ax2.set_title("Top 10 des variables importantes (locale)", fontsize=14)
                        ax2.set_xlim(-1, 1)  # 🔧 fixe les limites X entre -1 et +1
                        ax2.axvline(0, color='black', linewidth=0.5)
                        ax2.grid(True, axis='x', linestyle='--', alpha=0.3)
                        fig2.tight_layout()
                        
                        st.pyplot(fig2)
                
                        st.markdown("""
                        <style>
                            .description-text {
                                font-size: 14px;
                                color: white;
                            }
                        </style>
                        <div class="description-text">
                            Ce graphique représente l’impact des 10 variables les plus importantes pour ce client.
                            <br><br>
                            🟥 : contribution <b>positive</b> (le modèle prédit + risque)<br>
                            🟦 : contribution <b>négative</b> (le modèle prédit - risque)
                        </div>
                        """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Erreur dans l’interprétation SHAP : {e}")
                    
            #----------------------------------------------------------------------------------------------------------
            with st.expander("📉 Histogramme des probabilités (cliquez pour visualiser)", expanded=False):

                model_features = [col for col in df_client_with_target.columns if col not in ['TARGET', 'proba', 'PREDICTION']]  # features utiles

                # Prédiction des probabilités pour tous les clients si pas déjà fait
                if 'proba' not in df_client_with_target.columns:
                    X_all = df_client_with_target[model_features]
                    # Appliquer le scaler (MinMaxScaler) sur les données client
                    try:
                        X_scaled = scaler.transform(X_all)
                        X_scaled_df = pd.DataFrame(X_scaled, columns=model_features, index=df_client_with_target.index)
                    except Exception as e:
                        st.error(f"Erreur lors de la transformation des données : {e}")
                        st.stop()
                                
                df_client_with_target['proba'] = model.predict_proba(X_scaled_df)[:, 1]  # proba de refus
                
                # Prédiction binaire
                df_client_with_target['PREDICTION'] = (df_client_with_target['proba'] >= threshold).astype(int)
            
                # Récupération du score du client sélectionné
                client_score = df_client_with_target.loc[selected_id, 'proba']

                # Construction de l’histogramme
                fig = px.histogram(
                    df_client_with_target,
                    x='proba',
                    color='TARGET',
                    nbins=50,
                    barmode='overlay',
                    opacity=0.6,
                    color_discrete_map={0: 'green', 1: 'red'},
                    labels={'proba': 'Score de probabilité', 'TARGET': 'Classe réelle'}
                    #title="Distribution des scores de probabilité (modèle de scoring)"
                )
            
                # Ligne bleue : client sélectionné
                fig.add_vline(
                    x=client_score,
                    line_color="white",
                    line_width=2,
                    line_dash="dash",
                    annotation_text=f"Client<br>{selected_id}",
                    annotation_position="top right"
                )
            
                # Ligne rouge : seuil de décision
                fig.add_vline(
                    x=threshold,
                    line_color="red",
                    line_width=2,
                    line_dash="dot",
                    annotation_text=f"Seuil décision<br>{int(threshold*100)}%",
                    annotation_position="top right"
                )
            
                # Layout
                fig.update_layout(
                    height=500,
                    xaxis_title="Probabilité de défaillance",
                    yaxis_title="Nombre de clients",
                    legend_title="Classe réelle",
                    margin=dict(t=60, b=80),
                    xaxis_tickformat=".0%"
                )
            
                # Affichage dans Streamlit
                st.plotly_chart(fig, use_container_width=True)

            #----------------------------------------------------------------------------------------------------------
            with st.expander("🧮 Matrice de confusion (cliquez pour visualiser)", expanded=False):
                
                # Prédictions binaires selon le seuil
                y_true = df_client_with_target['TARGET']
                y_pred = (df_client_with_target['proba'] >= threshold).astype(int)
                
                # Calcul de la matrice de confusion
                cm = confusion_matrix(y_true, y_pred)
                
                tn, fp, fn, tp = cm.ravel()
                total = cm.sum()
                
                # 3. Texte annoté dans chaque cellule (valeur + % + label)
                z_text = [
                    [f"TN<br>{tn} ({tn/total:.1%})", f"FP<br>{fp} ({fp/total:.1%})"],
                    [f"FN<br>{fn} ({fn/total:.1%})", f"TP<br>{tp} ({tp/total:.1%})"]
                ]
                
                # 4. Valeurs numériques pour le heatmap (utilisées pour les couleurs)
                z_values = [[tn, fp], [fn, tp]]
                
                # 5. Affichage
                fig_cm = ff.create_annotated_heatmap(
                    z=z_values,
                    x=["0 (Non défaillant)", "1 (Défaillant)"],
                    y=["0 (Non défaillant)", "1 (Défaillant)"],
                    annotation_text=z_text,
                    colorscale="Blues",
                    showscale=True,
                    font_colors=["black"],  # pour garantir la lisibilité
                    hoverinfo="skip"
                )
                
                fig_cm.update_layout(
                    height=300,  # hauteur en pixels
                    width=400,   # largeur en pixels
                    #title_text=f"Matrice de confusion",
                    xaxis=dict(title="Prédiction"),
                    yaxis=dict(title="Classe réelle", autorange='reversed'),  # inverse l’ordre pour cohérence
                    margin=dict(t=50, l=100)
                )
                
                st.plotly_chart(fig_cm, use_container_width=True)
            
            #----------------------------------------------------------------------------------------------------------
            with st.expander("🕸️ Radar : Comparaison client vs Moyenne Décision (cliquez pour visualiser)", expanded=False):
                # Sélection dynamique des variables pour le radar
                features_radar = st.multiselect(
                        "Selectionnez les variables à afficher dans le Radar :",
                        options=available_features,
                        default=available_features[:3]  # ou choisis une sélection par défaut pertinente
                )
                    
                # Si rien sélectionné, afficher un message
                if not features_radar:
                    st.warning("Veuillez sélectionner au moins une variable.")
                    st.stop()
    
                # Clients ayant eu la même décision que celle prédite par l’API
                df_same_decision = df_client_with_target[df_client_with_target['TARGET'] == prediction]
                
                if df_same_decision.empty:
                    st.warning("Aucun autre client avec cette même décision.")
                else:
                   # Extraction sécurisée du vecteur client
                    if isinstance(client_data_with_target, pd.DataFrame):
                        client_vector = client_data_with_target[features_radar].iloc[0].to_frame().T
                    else:
                        client_vector = client_data_with_target[features_radar].to_frame().T
                    client_vector.index = ['Client']
            
                    # Concaténation avec les autres clients
                    radar_data = pd.concat([df_same_decision[features_radar], client_vector])
            
                    # Normalisation
                    scaler = MinMaxScaler()
                    radar_scaled = pd.DataFrame(scaler.fit_transform(radar_data), columns=features_radar, index=radar_data.index)
            
                    client_values = radar_scaled.loc['Client'].tolist()
                    avg_values = radar_scaled.drop('Client').mean().tolist()
            
                    # Fermeture de la boucle pour le radar
                    features_loop = features_radar + [features_radar[0]]
                    client_values += [client_values[0]]
                    avg_values += [avg_values[0]]

                    client_name = "Client "+selected_id+" accepté" if prediction == 0 else "Client "+selected_id+" refusé"
                    moy_name = "Moyenne des clients acceptés" if prediction == 0 else "Moyenne des clients refusés"
                    
                    # Affichage Plotly
                    fig = go.Figure()
            
                    fig.add_trace(go.Scatterpolar(
                        r=client_values,
                        theta=features_loop,
                        fill='toself',
                        name=client_name
                    ))
            
                    fig.add_trace(go.Scatterpolar(
                        r=avg_values,
                        theta=features_loop,
                        fill='toself',
                        name=moy_name
                    ))
            
                    fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1],
                            gridcolor='rgba(0, 0, 0, 0.5)',
                            gridwidth=0.5,
                            griddash='dot',
                            linecolor='rgba(0, 0, 0, 0.8)',
                            linewidth=0.5,
                            tickfont=dict(size=12, color='black', family='Arial'),
                            tickcolor='black',
                            ticklen=8,
                            tickwidth=2,
                            showticklabels=True
                        ),
                        angularaxis=dict(
                            tickfont=dict(size=10, color='white', family='Arial'),
                            tickcolor='white',
                            ticklen=10,
                            tickwidth=2,
                            rotation=45,        # décale les labels angulaires (si besoin)
                            direction="clockwise"
                        )
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.2,
                        xanchor="center",
                        x=0.5
                    ),
                    showlegend=True,
                    autosize=True,
                    margin=dict(t=30, b=50)
                    )
            
                    # Centrage dans Streamlit
                    col1, col2, col3 = st.columns([1, 10, 1])
                    with col2:
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown(f"""
                        <div style="font-size:14px; color:white;">
                            <p>Ce graphique permet de comparer visuellement les caractéristiques du client <b>{selected_id}</b> avec la moyenne des clients ayant obtenu la même décision que lui.</p>
                            <ul>
                                <li>Toutes les variables sont normalisées entre 0 et 1 pour permettre une comparaison cohérente malgré des échelles différentes </li>
                                <li>L’utilisateur peut observer les forces et faiblesses du client par rapport à ses pairs sur les variables sélectionnées.</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
            
            #----------------------------------------------------------------------------------------------------------
            with st.expander("🔎 Visualisation des caractéristiques clients (cliquez pour visualiser)", expanded=False):
                st.markdown("<span style='font-size:14px; color: white;'>Comparez la valeur d’une variable du client sélectionné à la distribution des autres clients.</span>", unsafe_allow_html=True)
                # Liste de toutes les features (colonnes)
                features = df_client.columns.tolist()

                with st.container():
                    col1, _ = st.columns([3, 1])
                    with col1:
                        # Sélecteur de variable
                        feature_selected = st.selectbox("Sélectionnez une variable :", features, key="feature_compare")
                
                # Valeur du client sélectionné
                client_value = df_client.loc[selected_id, feature_selected]
            
                # Création du graphique
                fig, ax = plt.subplots(figsize=(10, 3))
                sns.histplot(df_client[feature_selected], bins=30, kde=True, color="lightblue", ax=ax)
                ax.axvline(client_value, color='red', linewidth=2, label=f"Client {selected_id}")
                #ax.set_title(f"Distribution de '{feature_selected}'", fontsize=13)
                ax.set_xlabel(feature_selected)
                ax.set_ylabel('Nombres')
                # Récupérer les limites de l'axe Y pour placer la flèche verticalement au milieu
                y_min, y_max = ax.get_ylim()
                mid_y = y_max * 0.5
                
                # Annotation latérale (texte à droite, flèche vers la ligne)
                ax.annotate(
                    f"Client {selected_id}",
                    xy=(client_value, mid_y),                  # destination de la flèche (trait rouge)
                    xytext=(client_value + (df_client[feature_selected].max() * 0.05), mid_y),  # texte à droite
                    ha='left',
                    va='center',
                    color='red',
                    arrowprops=dict(facecolor='red', arrowstyle='->'),
                    fontsize=10,
                    fontweight='bold'
                )

                #ax.legend()
                st.pyplot(fig)
            
                st.markdown(f"""<span style='font-size:14px; color: red;'><b>|</b></span>
                <span style='font-size:14px; color: white;'> : Position du client <b>{selected_id}</b> pour <b>{feature_selected}</b> : <code>{client_value:.2f}</code>
                </span>
                """, unsafe_allow_html=True)
            #----------------------------------------------------------------------------------------------------------         
            with st.expander("📊 Analyse bi-variée entre deux variables (cliquez pour visualiser)", expanded=False):
                st.markdown("<span style='font-size:14px; color: white;'>Comparez 2 variables du client sélectionné à la distribution des autres clients.</span>", unsafe_allow_html=True)
                #st.markdown("<span style='font-size:14px; color: white;'>Sélectionnez 2 variables à comparer :</span>", unsafe_allow_html=True)
                
                selected_row = df_client.loc[selected_id]
                features = [col for col in df_client.columns if col != "SK_ID_CURR"]
                col1, col2 = st.columns(2)
                with col1:
                    x_var = st.selectbox("1ère variable (axe X) :", features, key="x_select")
                with col2:
                    y_var = st.selectbox("2ème variable (axe Y) :", features, key="y_select")
                
                is_x_cat = df_client[x_var].dtype == "object"
                is_y_cat = df_client[y_var].dtype == "object"
                
                fig, ax = plt.subplots(figsize=(10, 3))
                
                if not is_x_cat and not is_y_cat:
                    sns.scatterplot(data=df_client, x=x_var, y=y_var, ax=ax)
                    x = selected_row[x_var]
                    y = selected_row[y_var]
                    ax.scatter(x, y, color='red', s=40, zorder=5)
                
                   # ax.annotate(
                    #    f"Client {selected_id}\n{x_var} = {x:.2f}\n{y_var} = {y:.2f}",
                    #    xy=(x, y),
                     #   xytext=(x + 0.02 * df_client[x_var].std(), y + 0.02 * df_client[y_var].std()),
                     #   fontsize=9,
                     #   color='red',
                     #   arrowprops=dict(facecolor='red', arrowstyle='->'),
                     #   ha='left'
                    #)
                    #ax.set_title(f"{y_var} en fonction de {x_var}")
                
                elif is_x_cat and not is_y_cat:
                    sns.boxplot(data=df_client, x=x_var, y=y_var, ax=ax)
                    x = selected_row[x_var]
                    y = selected_row[y_var]
                    ax.scatter(x, y, color='red', s=40, zorder=5)
                
                   # ax.annotate(
                    #    f"Client {selected_id}\n{x_var} = {x}\n{y_var} = {y:.2f}",
                   #     xy=(x, y),
                   #     xytext=(0.2, y + 0.05 * df_client[y_var].std()),
                  #      textcoords='data',
                  #      fontsize=9,
                  #      color='red',
                  #      arrowprops=dict(facecolor='red', arrowstyle='->'),
                  #      ha='left'
                  #  )
                   # ax.set_title(f"{y_var} selon {x_var}")
                
                elif not is_x_cat and is_y_cat:
                    sns.boxplot(data=df_client, x=y_var, y=x_var, ax=ax)
                    x = selected_row[y_var]
                    y = selected_row[x_var]
                    ax.scatter(x, y, color='red', s=40, zorder=5)
                
                    #ax.annotate(
                    #    f"Client {selected_id}\n{y_var} = {x}\n{x_var} = {y:.2f}",
                    #    xy=(x, y),
                   #     xytext=(0.2, y + 0.05 * df_client[x_var].std()),
                    #    textcoords='data',
                    #    fontsize=9,
                    #    color='red',
                   #     arrowprops=dict(facecolor='red', arrowstyle='->'),
                   #     ha='left'
                   # )
                    #ax.set_title(f"{x_var} selon {y_var}")
                
                else:
                    # Catégorie vs Catégorie → heatmap
                    crosstab = pd.crosstab(df_client[x_var], df_client[y_var])
                    sns.heatmap(crosstab, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
                    #ax.set_title(f"Fréquence croisée : {x_var} vs {y_var}")
                    # Pas de point individuel possible ici
                
                st.pyplot(fig)
                # Affichage des valeurs en dehors du graphique :
                if x is not None and y is not None:
                    st.markdown(f"""
                    <div style='color: white; font-size:14px; margin-top: 10px;'>
                    <span style='font-size: 20px; color: red;'>●</span> : Position du client <b>{selected_id}</b> pour <b>{x_var}</b> = <code>{x}</code> et <b>{y_var}</b> = <code>{y}</code>
                    </div>
                    """, unsafe_allow_html=True)
                    

                
        else:
            st.error(f"Erreur API : {response.status_code}")
            st.json(response.json())

    except Exception as e:
        st.error(f"Erreur de connexion à l'API : {e}")