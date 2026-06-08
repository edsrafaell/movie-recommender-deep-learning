import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Flatten,
    Concatenate,
    Dense,
    Dropout
)

# ==========================
# Carregar Dataset
# ==========================

dados = pd.read_csv(
    "dados/u.data",
    sep="\t",
    names=["usuario", "filme", "nota", "timestamp"]
)

# Ajustar IDs para começar em 0
dados["usuario"] = dados["usuario"] - 1
dados["filme"] = dados["filme"] - 1

# Quantidades
num_usuarios = dados["usuario"].nunique()
num_filmes = dados["filme"].nunique()

print(f"Usuários: {num_usuarios}")
print(f"Filmes: {num_filmes}")

# ==========================
# Entradas e Saídas
# ==========================

X_usuario = dados["usuario"].values
X_filme = dados["filme"].values

y = dados["nota"].values.astype(np.float32)

# ==========================
# Divisão dos Dados
# ==========================

(
    usuario_treino,
    usuario_teste,
    filme_treino,
    filme_teste,
    y_treino,
    y_teste
) = train_test_split(
    X_usuario,
    X_filme,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Entradas da Rede
# ==========================

entrada_usuario = Input(shape=(1,))
entrada_filme = Input(shape=(1,))

# ==========================
# Embeddings
# ==========================

embedding_usuario = Embedding(
    input_dim=num_usuarios,
    output_dim=50
)(entrada_usuario)

embedding_filme = Embedding(
    input_dim=num_filmes,
    output_dim=50
)(entrada_filme)

# ==========================
# Flatten
# ==========================

usuario_flat = Flatten()(embedding_usuario)
filme_flat = Flatten()(embedding_filme)

# ==========================
# Combinar Informações
# ==========================

x = Concatenate()([usuario_flat, filme_flat])

# ==========================
# Rede Neural
# ==========================

x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)

x = Dense(64, activation="relu")(x)

saida = Dense(1)(x)

# ==========================
# Criar Modelo
# ==========================

modelo = Model(
    inputs=[entrada_usuario, entrada_filme],
    outputs=saida
)

# ==========================
# Compilar
# ==========================

modelo.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# ==========================
# Resumo
# ==========================

modelo.summary()

# ==========================
# Treinamento
# ==========================

historico = modelo.fit(
    [usuario_treino, filme_treino],
    y_treino,
    validation_split=0.2,
    epochs=20,
    batch_size=64
)


plt.figure(figsize=(10,5))

plt.plot(
    historico.history['loss'],
    label='Treino'
)

plt.plot(
    historico.history['val_loss'],
    label='Validação'
)

plt.title('Loss Treino x Validação')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()

plt.savefig('resultados/loss.png')

plt.show()

plt.figure(figsize=(10,5))

plt.plot(
    historico.history['mae'],
    label='MAE Treino'
)

plt.plot(
    historico.history['val_mae'],
    label='MAE Validação'
)

plt.title('MAE por Época')
plt.xlabel('Épocas')
plt.ylabel('MAE')
plt.legend()

plt.savefig('resultados/mae.png')

plt.show()

# ==========================
# Avaliação
# ==========================

loss, mae = modelo.evaluate(
    [usuario_teste, filme_teste],
    y_teste
)

rmse = np.sqrt(loss)

print("\n===== MÉTRICAS FINAIS =====")
print("MAE :", mae)
print("RMSE:", rmse)
print("LOSS:", loss)

# ==========================
# Salvar Modelo
# ==========================

modelo.save("modelos/recomendador.h5")

print("\nModelo salvo com sucesso!")
with open(
    "resultados/metricas.txt",
    "w",
    encoding="utf-8"
) as arquivo:

    arquivo.write(
        f"MAE: {mae}\n"
    )

    arquivo.write(
        f"RMSE: {rmse}\n"
    )

    arquivo.write(
        f"LOSS: {loss}\n"
    )
    
plt.figure(figsize=(6,4))

modelos = [
    "Baseline",
    "Deep Learning"
]

maes = [
    1.00,
    mae
]

plt.bar(
    modelos,
    maes
)

plt.title(
    "Comparação de MAE"
)

plt.ylabel(
    "MAE"
)

plt.savefig(
    "resultados/comparacao_mae.png"
)

plt.show()