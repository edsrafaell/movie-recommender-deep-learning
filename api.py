from flask import Flask, jsonify, render_template

import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

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
# Carregar avaliações
# ==========================

dados = pd.read_csv(
    "dados/u.data",
    sep="\t",
    names=["usuario", "filme", "nota", "timestamp"]
)

# ==========================
# ENDPOINT
# ==========================

@app.route("/recomendar/<int:usuario>")
def recomendar(usuario):

    usuario_id = usuario - 1

    # ==========================
    # Filmes favoritos do usuário
    # ==========================

    historico_usuario = dados[
        dados["usuario"] == usuario
    ]

    top_gostou = historico_usuario.sort_values(
        by="nota",
        ascending=False
    ).head(5)

    top_gostou = top_gostou.merge(
        filmes,
        left_on="filme",
        right_on="filme_id"
    )

    favoritos = []

    for _, filme in top_gostou.iterrows():

        favoritos.append({
            "titulo": filme["titulo"],
            "nota": int(filme["nota"])
        })

    # ==========================
    # Gerar recomendações
    # ==========================

    todos_filmes = filmes["filme_id"].values

    usuarios = np.full(
        len(todos_filmes),
        usuario_id
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
            "titulo": filme["titulo"],
            "nota_prevista": round(
                min(5.0, float(filme["nota_prevista"])),
                2
        )
    })

    # ==========================
    # Resposta JSON
    # ==========================

    return jsonify({
        "usuario": usuario,
        "criterio": (
            "As recomendações foram geradas com base "
            "nas avaliações anteriores do usuário."
        ),
        "filmes_favoritos": favoritos,
        "recomendacoes": resultado
    })

# ==========================
# Iniciar API
# ==========================

if __name__ == "__main__":
    app.run(debug=True)