import joblib

# SEUIL_OPTIMAL = 0.6
# Chargement du seuil optimal depuis un fichier .txt
with open("threshold.txt", "r") as f:
    SEUIL_OPTIMAL = float(f.read().strip())

def load_model(path="LGBMClm_model.pkl"):
    return joblib.load(path)

def predict_score(model, client_data, seuil=SEUIL_OPTIMAL):
    proba = model.predict_proba([client_data])[0][1]
    decision = "Refusé" if proba >= seuil else "Accepté"

    return {
        "score_proba": round(proba, 3),
        "seuil": seuil,
        "decision": decision
    }