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

#st.image("https://user.oc-static.com/upload/2023/03/22/16794938722698_Data%20Scientist-P7-01-banner.png", use_container_width=True)
st.markdown(
    """
    <div style="text-align: center;"><img src="https://user.oc-static.com/upload/2023/03/22/16794938722698_Data%20Scientist-P7-01-banner.png" width="75%"></div>
    """, unsafe_allow_html=True
)
#st.title("Dashboard de credit scoring")
st.markdown(
    "<h1 style='text-align: center;'>Dashboard de credit scoring</h1>",
    unsafe_allow_html=True
)

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

    # Pas de titre dans la figure elle-même
    fig.update_layout(height=200, margin=dict(t=20, b=20, l=20, r=20)
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
client_ids, df_client, df_client_with_target, df = load_client_ids()

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

        if response.status_code == 200:
            result = response.json()
            proba = result.get("score_proba")
            decision = result.get("décision")
            st.markdown(f"<h1 style='text-align: center; margin-top: 0px; margin-bottom: 0px; font-size: 20px;'>Probabilité de défaillance du client { selected_id} (en %)</h1>", unsafe_allow_html=True)
            
            show_gauge(proba)  #Affiche la jauge ici

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

            # Affichage des données du client
            scaler = pipeline.named_steps['scaler']
            model = pipeline.named_steps['model']
            
            # Appliquer le scaler (MinMaxScaler) sur les données client
            try:
                X_scaled = scaler.transform(client_data)
                X_scaled_df = pd.DataFrame(X_scaled, columns=client_data.columns, index=client_data.index)
            except Exception as e:
                st.error(f"Erreur lors de la transformation des données : {e}")
                st.stop()
            
            # Prédiction de la probabilité
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
    
            # SHAP TreeExplainer
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
                    color_discrete_map={0: '#008BFB', 1: '#D83E69'},
                    labels={'proba': 'Score de probabilité', 'TARGET': 'Classe réelle'}
                    #title="Distribution des scores de probabilité (modèle de scoring)"
                )
            
                # Ligne bleue : client sélectionné
                fig.add_vline(
                    x=client_score,
                    line_color="red",
                    line_width=2,
                    line_dash="solid",
                    annotation=dict(
                        text=f"Client<br>{selected_id}",
                        font=dict(color="red", size=12),
                        showarrow=False,
                        xanchor="left",
                        yanchor="top"
                    )
                )
            
                # Ligne rouge : seuil de décision
                fig.add_vline(
                    x=threshold,
                    line_color="white",
                    line_width=2,
                    line_dash="solid",
                    annotation=dict(
                        text=f"Seuil décision<br>{int(threshold*100)}%",
                        font=dict(color="white", size=12),
                        showarrow=False,
                        xanchor="right",
                        yanchor="top"
                    )
                )
            
                # Layout
                fig.update_layout(
                    height=500,
                    xaxis_title="Probabilité de défaillance",
                    yaxis_title="Nombre de clients",
                    #legend_title="Classe réelle",
                    margin=dict(t=60, b=80),
                    xaxis_tickformat=".0%",
                    showlegend=False
                )

                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=1.00, y=-0.20,
                    text="🟥 : client défaillant<br>🟦 : client non défaillant",
                    showarrow=False,
                    align="left",
                    font=dict(size=12),
                    bordercolor="lightgray",
                    borderwidth=0.5
                )
                
                # Affichage dans Streamlit
                st.plotly_chart(fig, use_container_width=True)

                # Phrase d’interprétation dynamique
                if client_score < threshold:
                    risk_level = "faible"
                else:
                    risk_level = "élevé"
                
                # calcul le pourcentage de clients ayant un score plus faible que celui du client sélectionné, donc des clients moins risqués.
                percentile = (df_client_with_target["proba"] < client_score).mean() * 100

                st.markdown(f"""<span style='font-size:14px'>Cet histogramme montre comment se répartissent les clients selon leur probabilité de défaillance prédite par le modèle</span>
             """, unsafe_allow_html=True)

                st.markdown(f"""<div style="font-size:14px; color:white;">
                                <p>Résultat pour le client <b>{selected_id}</b> selectionné :</p>
                                <ul>
                                    <li>Le client a une probabilité de défaillance de <code>{client_score*100:.1f} %</code></li>
                                    <li>Cela correspond à un risque <code>{risk_level}</code>  selon le modèle</li>
                                    <li>Ce score le place dans les <code>{100 - percentile:.0f} %</code>  des clients les plus risqués</li>
                                    </ul>
                            </div>
                  """, unsafe_allow_html=True)
            

            #----------------------------------------------------------------------------------------------------------
            with st.expander("🧮 Matrice de confusion (cliquez pour visualiser)", expanded=False):
                
                # Prédictions binaires selon le seuil
                y_true = df_client_with_target['TARGET']
                y_pred = (df_client_with_target['proba'] >= threshold).astype(int)
                
                # Matrice de confusion
                cm = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel()
                total = cm.sum()
                
                # Texte dans chaque cellule (valeurs + pourcentage + type)
                z_text = [
                    [f"TN<br>{tn} ({tn/total:.1%})", f"FP<br>{fp} ({fp/total:.1%})"],
                    [f"FN<br>{fn} ({fn/total:.1%})", f"TP<br>{tp} ({tp/total:.1%})"]
                ]
                
                # Valeurs numériques pour le heatmap
                z_values = [[tn, fp], [fn, tp]]
                
                # 5. Création du heatmap Plotly
                fig_cm = ff.create_annotated_heatmap(
                    z=z_values,
                    x=["client non défaillant", "client défaillant"],  # prédiction
                    y=["client non défaillant", "client défaillant"],  # classe réelle
                    annotation_text=z_text,
                    #colorscale="Blues",
                    colorscale=[[0.0, "#e6f2ff"],  # bleu très clair (pour TN si valeur faible)
                    [0.25, "#cce0ff"],
                    [0.5, "#99c2ff"],
                    [0.75, "#4da6ff"],
                    [1.0, "#0066cc"]],
                    showscale=True,
                    font_colors=["black"],
                    hoverinfo="skip"
                )
                
                # Mise en forme générale
                fig_cm.update_layout(
                    height=300,
                    width=400,
                    xaxis=dict(title="Catégorie prédite par le modèle"),
                    yaxis=dict(title="Catégorie réelle", autorange='reversed'),
                    margin=dict(t=50, l=100)
                )
                
                # Mettre en valeur le client sélectionné
                y_true_client = df_client_with_target.loc[selected_id, 'TARGET'] # 0 ou 1
                y_pred_client = int(df_client_with_target.loc[selected_id, 'proba'] >= threshold)  # 0 ou 1
                client_score = df_client_with_target.loc[selected_id, 'proba']
                
                # Encadré rouge autour de la cellule correspondante
                fig_cm.add_shape(
                    type="rect",
                    x0=y_pred_client - 0.5,
                    x1=y_pred_client + 0.5,
                    y0=y_true_client - 0.5,
                    y1=y_true_client + 0.5,
                    line=dict(color="red", width=3)
                )
                
                # Annotation dans la cellule
                x_labels = ["client non défaillant", "client défaillant"]
                y_labels = ["client non défaillant", "client défaillant"]
                x_label = x_labels[y_pred_client]
                y_label = y_labels[y_true_client]
                
                # Légende personnalisée en bas à droite
                fig_cm.add_annotation(
                    xref="paper", yref="paper",
                    x=1.02, y=-0.15,
                    text=f"<span style='color:red; font-weight:bold;'>▢ Client {selected_id}</span>",
                    showarrow=False,
                    font=dict(size=12)
                )
                
                # Affichage dans Streamlit
                st.plotly_chart(fig_cm, use_container_width=True)
                
                st.markdown(f"""<span style='font-size:14px'>La matrice de confusion sert à évaluer la qualité des prédictions du modèle de classification comme suit :</span>
                <br>
                <table style="width:100%; border: 0.5px solid #ccc; border-collapse: collapse;font-size: 12px;">
                  <thead>
                    <tr>
                      <th style="border: 0.5px solid #ccc; padding: 6px;">Cas</th>
                      <th style="border: 0.5px solid #ccc; padding: 6px;">Nom</th>
                      <th style="border: 0.5px solid #ccc; padding: 6px;">Interprétation</th>
                      <th style="border: 0.5px solid #ccc; padding: 6px;">Impact métier</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td style="border: 0.5px solid #ccc; padding: 6px;">TN</td>
                      <td style="border: 0.5px solid #ccc; padding: 6px;">Vrais négatifs</td>
                      <td style="border: 0.5px solid #ccc; padding: 6px;">Le modèle a bien identifié un client non défaillant</td>
                      <td style="border: 0.5px solid #ccc; padding: 6px; color:#00FF7F;">✅ OK (client accepté à juste titre)</td>
                    </tr>
                    <tr>
                      <td style="border: 0.5px solid #ccc; padding: 6px;">TP</td>
                      <td style="border: 0.5px solid #ccc; padding: 6px;">Vrais positifs</td>
                      <td style="border: 0.5px solid #ccc; padding: 6px;">Le modèle a bien identifié un client défaillant</td>
                      <td style="border: 0.5px solid #ccc; padding: 6px; color:#00FF7F;">✅ OK (client risqué à juste titre)</td>
                    </tr>
                    <tr>
                      <td style="border: 0.5px solid #ccc; padding: 6px;">FP</td>
                      <td style="border: 0.5px solid #ccc; padding: 6px;">Faux positifs</td>
                      <td style="border: 0.5px solid #ccc; padding: 6px;">Le modèle a refusé un client non défaillant par erreur</td>
                      <td style="border: 0.5px solid #ccc; padding: 6px; color:#D83E69;">❌ Mauvais pour le client (perte d'opportunité)</td>
                    </tr>
                    <tr>
                      <td style="border: 0.5px solid #ccc; padding: 6px;">FN</td>
                      <td style="border: 0.5px solid #ccc; padding: 6px;">Faux négatifs</td>
                      <td style="border: 0.5px solid #ccc; padding: 6px;">Le modèle a accepté un client défaillant</td>
                      <td style="border: 0.5px solid #ccc; padding: 6px; color:#D83E69;">❌ Mauvais pour l’entreprise (perte financière)</td>
                    </tr>
                  </tbody>
                </table></span>
                """, unsafe_allow_html=True)
                # Identifier la catégorie (TN, TP, FP, FN)
                if y_true_client == 0 and y_pred_client == 0:
                    cat = "TN"
                    label = "Vrais négatifs"
                    interpretation = "Le modèle a correctement prédit que le client sera non défaillant"
                elif y_true_client == 1 and y_pred_client == 1:
                    cat = "TP"
                    label = "Vrais positifs"
                    interpretation = "Le modèle a correctement prédit que le client sera défaillant"
                elif y_true_client == 0 and y_pred_client == 1:
                    cat = "FP"
                    label = "Faux positifs"
                    interpretation = "Le modèle a prédit une défaillance à tort pour ce client qui était sain"
                elif y_true_client == 1 and y_pred_client == 0:
                    cat = "FN"
                    label = "Faux négatifs"
                    interpretation = "Le modèle n'a pas détecté une vraie défaillance de ce client à risque"

                st.markdown(f"""<div style="font-size:14px; color:white;">
                                <p>Résultat pour le client <b>{selected_id}</b> selectionné :</p>
                                <ul>
                                    <li>Le client a été catégorisé en <b><code>{cat}</code></b> :  <code>{label}</code></li>
                                    <li>{interpretation}</li>
                                    </ul>
                            </div>
                  """, unsafe_allow_html=True)
   
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
                client_value = df_client_with_target.loc[selected_id, feature_selected]
            
                # Création du graphique
                fig, ax = plt.subplots(figsize=(10, 3))

                # Histogrammes séparés par classe TARGET
                sns.histplot(data=df_client_with_target, x=feature_selected, hue="TARGET", hue_order=[0, 1], bins=30, kde=True, palette={0: "#008BFB", 1: "#D83E69"}, alpha=0.5, ax=ax)
          
                # Ligne verticale pour le client sélectionné
                ax.axvline(client_value, color='red', linewidth=2, label=f"Client {selected_id}")
                
                ax.set_xlabel(feature_selected)
                ax.set_ylabel('Nombre')
                
                # Limites pour annotation
                y_min, y_max = ax.get_ylim()
                mid_y = y_max * 0.5
                
                # Annotation latérale (texte à droite, flèche vers la ligne)
                ax.annotate(
                    f"Client {selected_id}",
                    xy=(client_value, y_max),
                    xytext=(client_value + (df_client_with_target[feature_selected].max() * 0.05), y_max),
                    ha='center',
                    va='bottom',
                    color='red',
                    #arrowprops=dict(facecolor='#00FA9A', arrowstyle='->'),
                    fontsize=10,
                    fontweight='bold'
                )
                
                # Légende personnalisée
                # 🛠 Supprimer légende auto-générée
                ax.legend_.remove()
                
                # 🛠 Recréer manuellement la légende avec les bons labels
                from matplotlib.lines import Line2D
                custom_legend = [
                    Line2D([0], [0], color="#008BFB", lw=2, label="Client non défaillant"),
                    Line2D([0], [0], color="#D83E69", lw=2, label="Client défaillant")
                ]
                ax.legend(handles=custom_legend)

                st.pyplot(fig)
            
                st.markdown(f"""<span style='font-size:14px; color: red;'><b>|</b></span>
                <span style='font-size:14px; color: white;'> : Position du client <b>{selected_id}</b> pour <b>{feature_selected}</b> : <code>{client_value:.2f}</code>
                </span>
                """, unsafe_allow_html=True)

            #----------------------------------------------------------------------------------------------------------         
            with st.expander("📊 Analyse bi-variée entre deux variables (cliquez pour visualiser)", expanded=False):
                st.markdown("<span style='font-size:14px; color: white;'>Comparez 2 variables du client sélectionné à la distribution des autres clients.</span>", unsafe_allow_html=True)
                #st.markdown("<span style='font-size:14px; color: white;'>Sélectionnez 2 variables à comparer :</span>", unsafe_allow_html=True)
                
                selected_row = df_client_with_target.loc[selected_id]
                features = [col for col in df_client_with_target.columns if col not in ("SK_ID_CURR","TARGET")]
                col1, col2 = st.columns(2)
                with col1:
                    x_var = st.selectbox("1ère variable (axe X) :", features, key="x_select")
                with col2:
                    y_var = st.selectbox("2ème variable (axe Y) :", features, key="y_select")
                
                is_x_cat = df_client_with_target[x_var].dtype == "object"
                is_y_cat = df_client_with_target[y_var].dtype == "object"
                
                fig, ax = plt.subplots(figsize=(10, 3))
                
                if not is_x_cat and not is_y_cat:
                    palette = {0: "#008BFB", 1: "#D83E69"}
       
                    sns.scatterplot(data=df_client_with_target, x=x_var, y=y_var, hue="TARGET", palette=palette, ax=ax, alpha=0.6)
                    # Modifier les labels de la légende
                    handles, labels = ax.get_legend_handles_labels()
                    labels = ["Client non défaillant" if l=="0" else "Client défaillant" for l in labels]
                    ax.legend(handles=handles, labels=labels)

                    x = selected_row[x_var]
                    y = selected_row[y_var]
                    ax.scatter(x, y, color='#00FF7F', s=50, zorder=5)
                
                elif is_x_cat and not is_y_cat:
                    sns.boxplot(data=df_client_with_target, x=x_var, y=y_var, ax=ax)
                    x = selected_row[x_var]
                    y = selected_row[y_var]
                    ax.scatter(x, y, color='#00FF7F', s=50, zorder=5)
                
                elif not is_x_cat and is_y_cat:
                    sns.boxplot(data=df_client_with_target, x=y_var, y=x_var, ax=ax)
                    x = selected_row[y_var]
                    y = selected_row[x_var]
                    ax.scatter(x, y, color='#00FF7F', s=50, zorder=5)
                
                else:
                    # Catégorie vs Catégorie → heatmap
                    crosstab = pd.crosstab(df_client_with_target[x_var], df_client[y_var])
                    sns.heatmap(crosstab, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
                
                st.pyplot(fig)
                # Affichage des valeurs en dehors du graphique :
                if x is not None and y is not None:
                    st.markdown(f"""
                    <div style='color: white; font-size:14px; margin-top: 10px;'>
                    <span style='font-size: 20px; color: #00FF7F;'>●</span> : Position du client <b>{selected_id}</b> pour <b>{x_var}</b> = <code>{x}</code> et <b>{y_var}</b> = <code>{y}</code>
                    </div>
                    """, unsafe_allow_html=True)
                    
            #----------------------------------------------------------------------------------------------------------          
            with st.expander("🧪 Simulation du score de probabilité (cliquez pour visualiser)", expanded=False):
                st.markdown(f"""<span style='font-size:14px; color: white;'>Modifiez les 6 variables ci-dessous pour observer l’impact sur la probabilité de défaillance du client <b>{selected_id}</b> :</span>""", unsafe_allow_html=True)
            
                # Supprimer les colonnes non utilisées
                cols_to_drop = ["TARGET", "proba", "PREDICTION", "SK_ID_CURR"]
                df_features = [col for col in df_client_with_target.columns if col not in cols_to_drop]
            
                # Vérification du nombre de colonnes
                model_features = pipeline.named_steps['model'].feature_name_
            
                if len(df_features) != len(model_features):
                    st.error(f"❌ Incohérence : {len(df_features)} colonnes dans df_client_with_target vs {len(model_features)} dans le modèle.")
                else:
                    # Mapping complet : vrai nom ➜ nom du modèle
                    full_feature_mapping = dict(zip(df_features, model_features))
            
                    # Liste des variables modifiables
                    vars_to_modify = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "PAYMENT_RATE", "CODE_GENDER", "DAYS_EMPLOYED"]
            
                    # Valeurs actuelles du client sélectionné
                    original_row = df_client_with_target.loc[[selected_id]]
                    original_values = original_row.loc[selected_id, vars_to_modify]
                    #st.write("original_row:",original_row)
                    #st.write("original_values:",original_values)       
                    
                    with st.form(key="form_simulation"):
                                        
                        col1, col2, col3 = st.columns(3)
                        EXT_SOURCE_1_real = col1.slider("EXT_SOURCE_1", 0.0, 1.0, float(original_values["EXT_SOURCE_1"]), step=0.001, format="%.3f", help="Score externe basé sur la stabilité professionnelle du client")
                        EXT_SOURCE_2_real = col2.slider("EXT_SOURCE_2", 0.0, 1.0, float(original_values["EXT_SOURCE_2"]), step=0.001, format="%.3f", help="Score externe basé sur l'historique bancaire du client")
                        EXT_SOURCE_3_real = col3.slider("EXT_SOURCE_3", 0.0, 1.0, float(original_values["EXT_SOURCE_3"]), step=0.001, format="%.3f", help="Score externe basé sur l’analyse comportementale du client")
            
                        col4, col5, col6 = st.columns(3)
                        PAYMENT_RATE_real = col4.slider("PAYMENT_RATE (%)", 0.0, 100.0, float(original_values["PAYMENT_RATE"] * 100), step=0.01, format="%.2f", help="Montant de remboursement mensuel rapporté au crédit (en %)") / 100
                        DAYS_EMPLOYED_real = col5.slider("DAYS_EMPLOYED", 0, 20000, round(original_values["DAYS_EMPLOYED"]), step=1, format="%.0f", help="Nombre de jours depuis le début de l’emploi (jusqu’à 20 000 jours)") 
                        gender_map = {0: "Male", 1: "Female", 0.0: "Male", 1.0: "Female", "0": "Male", "1": "Female"}
                        gender_str = gender_map.get(original_values["CODE_GENDER"], "M") 
                        CODE_GENDER = col6.selectbox(
                                "CODE_GENDER",
                                ["Male", "Female"],
                                index=["Male", "Female"].index(gender_str),  # garde la valeur par défaut du client
                                help="Sexe du client"
                            )
            
                        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

                        with col3:
                            submitted = st.form_submit_button("🔄 Recalculer le score")
    
                        #submitted = st.form_submit_button("🔄 Recalculer le score")
            
                    if submitted:
                        # Mise à jour des valeurs modifiées dans la ligne d'origine
                        modified_row = original_row.copy()
                        modified_row.loc[selected_id, "EXT_SOURCE_1"] = EXT_SOURCE_1_real
                        modified_row.loc[selected_id, "EXT_SOURCE_2"] = EXT_SOURCE_2_real
                        modified_row.loc[selected_id, "EXT_SOURCE_3"] = EXT_SOURCE_3_real
                        modified_row.loc[selected_id, "PAYMENT_RATE"] = PAYMENT_RATE_real
                        modified_row.loc[selected_id, "CODE_GENDER"] = CODE_GENDER
                        modified_row.loc[selected_id, "DAYS_EMPLOYED"] = DAYS_EMPLOYED_real

                        # Préparation des données pour le modèle
                        X_old = original_row[df_features].copy()
                        X_new = modified_row[df_features].copy()
                
                        # Encodage de CODE_GENDER : M → 0, F → 1
                        X_old["CODE_GENDER"] = original_values["CODE_GENDER"]
                        X_new["CODE_GENDER"] = 0 if CODE_GENDER == "Male" else 1
                        
                        # Prédictions via pipeline (scaling + modèle)
                        old_score = pipeline.predict_proba(X_old)[0][1]
                        new_score = pipeline.predict_proba(X_new)[0][1]


                        # Affichage
                        if old_score <= threshold:
                            icone_response_old_score = "&#x2714;" #"&#x2705;"
                            color_reponse_old_score = "#008BFB" #49C289
                            decision_old_score = "Accepté"
                            if old_score <= (threshold/2):
                                interpretation_old_score = "🟩 Risque faible de défaillance"
                            else:
                                interpretation_old_score = "🟨 Risque modéré de défaillance"
                        if old_score > threshold:
                            icone_response_old_score = "&#x274C;"
                            color_reponse_old_score = "#D83E69"
                            decision_old_score = "Refusé"
                            if old_score > (threshold + ((1-threshold)/2)):
                                interpretation_old_score = "🟥 Risque très elevé de défaillance"
                            else:
                                interpretation_old_score = "🟧 Risque elevé de défaillance"
                        if new_score <= threshold:
                            icone_response_new_score = "&#x2714;" #"&#x2705;"
                            color_reponse_new_score = "#008BFB" #49C289
                            decision_new_score = "Accepté"
                            if new_score <= (threshold/2):
                                interpretation_new_score = "🟩 Risque faible de défaillance"
                            else:
                                interpretation_new_score = "🟨 Risque modéré de défaillance"
                        if new_score > threshold:
                            icone_response_new_score = "&#x274C;"
                            color_reponse_new_score = "#D83E69"
                            decision_new_score = "Refusé"
                            if new_score > (threshold + ((1-threshold)/2)):
                                interpretation_new_score = "🟥 Risque très elevé de défaillance"
                            else:
                                interpretation_new_score = "🟧 Risque elevé de défaillance"

                        initial_proba = old_score  
                        new_proba = new_score      
                        initial_decision = decision_old_score
                        new_decision = decision_new_score
                        
                        st.markdown(
                            f"""
                            <style>
                            .result-table {{
                                border-collapse: collapse;
                                font-size: 16px;
                                color: white;
                                width: 100%;
                                margin: 10px auto; /* Centrage horizontal */
                            }}
                            .result-table th, .result-table td {{
                                border: 1px solid #444;
                                padding: 5px;
                                text-align: center;
                            }}
                            .result-table th {{
                                background-color: #333;
                            }}
                            .result-table tr:nth-child(even) {{
                                background-color: #222;
                            }}
                            .highlighted-left {{
                                border-left: 2px solid white !important;
                                border-top: 2px solid white !important;
                                border-bottom: 2px solid white !important;
                            }}
                            .highlighted-mid {{
                                border-top: 2px solid white !important;
                                border-bottom: 2px solid white !important;
                            }}
                            .highlighted-right {{
                                border-right: 2px solid white !important;
                                border-top: 2px solid white !important;
                                border-bottom: 2px solid white !important;
                            }}
                            </style>
                            <div style="font-size:14px; color:white;"><p>Résultat de la simulation pour le client <b>{selected_id}</b> selectionné :</p></div>
                            <table class="result-table">
                                <tr>
                                    <th></th>
                                    <th><span style="font-size:13px; color:white;">Probabilité de défaillance</span></th>
                                    <th><span style="font-size:13px; color:white;">Intérprétation</span></th>
                                    <th><span style="font-size:13px; color:white;">Décision</span></th>
                                </tr>
                                <tr>
                                    <td style='text-align:left;'><span style="font-size:13px; color:white;"><b>Avec les valeurs initiales</b></span></td>
                                    <td><span style="font-size:13px; color:white;">{initial_proba:.1%}</span></td>
                                    <td style='text-align:left;'><span style="font-size:13px; color:white;">{interpretation_old_score}</span></td>
                                    <td style='text-align:left;'><b><span style="font-size:13px; color:{color_reponse_old_score}">{decision_old_score}</span></b> {icone_response_old_score}</font></td>
                                </tr>
                                <tr>
                                    <td class="highlighted-left" style='text-align:left;'><span style="font-size:13px; color:white;"><b>Avec les nouvelles valeurs</b></span></td>
                                    <td class="highlighted-mid"><span style="font-size:13px; color:white;">{new_proba:.1%}</span></td>
                                    <td class="highlighted-mid" style='text-align:left;'><span style="font-size:13px; color:white;">{interpretation_new_score}</span></td>
                                    <td class="highlighted-right" style='text-align:left;'><b><span style="font-size:13px; color:{color_reponse_new_score}">{decision_new_score}</font></b> {icone_response_new_score}</font></td>
                                </tr>
                            </table>
                            """,
                            unsafe_allow_html=True
                        )
                
        else:
            st.error(f"Erreur API : {response.status_code}")
            st.json(response.json())

    except Exception as e:
        st.error(f"Erreur de connexion à l'API : {e}")