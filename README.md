# API REST
O projeto disponibiliza uma API para consultar recomendações de filmes geradas pelo modelo de Deep Learning.

* Instruções de instalação
Instlação das bibliotecas
pip install pandas numpy tensorflow scikit-learn matplotlib

* Clonando o repositório:
Execute no terminal
git clone https://github.com/edsrafaell/movie-recommender-deep-learning
cd movie-recommender-deep-learning

* Criando ambiente virtual:
Execute no terminal
python -m venv venv

* Ativando o ambiente virtual:
Execute no terminal
venv\Scripts\activate

(VERIFICAÇÃO)
* Verificar a estrutura do projeto
movie-recommender-deep-learning/

├── dados/
│   ├── u.data
│   └── u.item
│
├── modelos/
│   └── recomendador.h5
│
├── templates/
│   └── index.html
│
├── resultados/
│   ├── loss.png
│   ├── mae.png
│   └── comparacao_mae.png
│
├── treino.py
├── api.py
├── requirements.txt
└── README.md

* Executando a API
Execute no terminal
python api.py

"Se tiver correro, o resultado sera:"
Running on http://127.0.0.1:5000

"Se der certo, abirar uma interface web"

* Abra o navegador e acesse
http://127.0.0.1:5000

* Consumir a API diretamente: Sem interface
http://127.0.0.1:5000/recomendar/ID_USUARIO
exemplo:
http://127.0.0.1:5000/recomendar/25

* Exemplo de resposta JSON
{
    "usuario": 25,
    "criterio": "As recomendações foram geradas com base nas avaliações anteriores do usuário.",
    "filmes_favoritos": [
    {
        "titulo": "GoodFellas (1990)",
        "nota": 5
    }
    ],
    "recomendacoes": [
    {
        "titulo": "Bitter Moon (1992)",
        "nota_prevista": 5.26
    }
    ]
}

* Como a API funciona
    O usuário informa um ID.
    A API consulta o histórico de avaliações desse usuário.
    Os 5 filmes mais bem avaliados são identificados.
    O modelo de Deep Learning gera uma previsão de nota para todos os filmes do catálogo.
    Os filmes com maiores notas previstas são selecionados.
    A API retorna os resultados em formato JSON.
    A interface web apresenta os filmes favoritos e as recomendações.

* Fluxo geral do sistema
Usuário
    ↓
Interface Web
    ↓
API Flask
    ↓
Modelo Deep Learning (.h5)
    ↓
Previsão das Notas
    ↓
Top 5 Recomendações
    ↓
Resposta JSON
    ↓
Exibição na Tela