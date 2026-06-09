from flask import Flask, jsonify

import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model

app = Flask(__name__)

# ==========================
# Carregar modelo
# ==========================

modelo = load_model(
    "modelos/recomendador.h5",
    compile=False
)

# ==========================
# Carregar filmes
# ==========================

filmes = pd.read_csv(
    "dados/u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=["filme_id", "titulo"]
)

filmes["filme_id"] = filmes["filme_id"] - 1

# ==========================
# ENDPOINT
# ==========================

@app.route("/recomendar/<int:usuario>")
def recomendar(usuario):

    usuario = usuario - 1

    todos_filmes = filmes["filme_id"].values

    usuarios = np.full(
        len(todos_filmes),
        usuario
    )

    previsoes = modelo.predict(
        [usuarios, todos_filmes],
        verbose=0
    )

    filmes["nota_prevista"] = previsoes

    top = filmes.sort_values(
        by="nota_prevista",
        ascending=False
    ).head(5)

    resultado = []

    for _, filme in top.iterrows():

        resultado.append({
            "filme": filme["titulo"],
            "nota_prevista": round(
                float(filme["nota_prevista"]),
                2
            )
        })

    return jsonify({
        "usuario": usuario + 1,
        "recomendacoes": resultado
    })

# ==========================
# Iniciar API
# ==========================

if __name__ == "__main__":
    app.run(debug=True)