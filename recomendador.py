import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model

# ==========================
# Carregar modelo treinado
# ==========================

modelo = load_model(
    "modelos/recomendador.h5",
    compile=False
)
# ==========================
# Carregar dados
# ==========================

filmes = pd.read_csv(
    "dados/u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=["filme_id", "titulo"]
)

dados = pd.read_csv(
    "dados/u.data",
    sep="\t",
    names=["usuario", "filme", "nota", "timestamp"]
)

# Ajustar IDs para começar em 0
filmes["filme_id"] = filmes["filme_id"] - 1

# ==========================
# Escolher usuário
# ==========================

usuario_id = int(input("Digite o ID do usuário (1-943): ")) - 1

# ==========================
# Gerar previsões
# ==========================

todos_filmes = filmes["filme_id"].values

usuarios = np.full(len(todos_filmes), usuario_id)

notas_previstas = modelo.predict(
    [usuarios, todos_filmes],
    verbose=0
)

filmes["nota_prevista"] = notas_previstas

# Top 5 recomendações
top_filmes = filmes.sort_values(
    by="nota_prevista",
    ascending=False
).head(5)

print("\n" + "="*50)
print(" SISTEMA DE RECOMENDAÇÃO DE FILMES ")
print("="*50)

print(f"\nUsuário ID: {usuario_id + 1}")

print("\nTop 5 Recomendações:\n")

for i, (_, filme) in enumerate(top_filmes.iterrows(), start=1):

    print(f"{i}. {filme['titulo']}")
    print(f"   Nota Prevista: {filme['nota_prevista']:.2f}\n")

print("="*50)