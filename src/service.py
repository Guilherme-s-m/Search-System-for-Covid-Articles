import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import torch
from torch import nn, optim, cosine_similarity
from sentence_transformers import SentenceTransformer
import dill




class EmbeddingAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EmbeddingAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = self.decoder(x)
        return x

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Dimensão dos embeddings do modelo SBERT escolhido
HIDDEN_DIM = 128     # Dimensão da camada intermediária do autoencoder
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001



def recommend_papers_from_csv(user_interest):
    model = SentenceTransformer(MODEL_NAME)
    document_embeddings = torch.load("refined_embeddings.pt")
    df = pd.read_csv("../data/metadata.csv")
    documents = df.to_dict(orient='records')
    # Gerar embedding para a consulta do usuário

    user_embedding = model.encode([user_interest], convert_to_numpy=True)
   
    autoencoder = EmbeddingAutoencoder(input_dim=384, hidden_dim=128)

    # Carregue os pesos salvos no modelo
    autoencoder.load_state_dict(torch.load("autoencoder_model.pth"))
    autoencoder.eval() 

    # Ajustar a dimensão do embedding da consulta usando o autoencoder, se fornecido
    user_embedding = torch.tensor(user_embedding, dtype=torch.float32)
    if autoencoder:
        user_embedding = autoencoder.encoder(user_embedding).detach()
    
    # Garantir que os embeddings dos documentos estejam no formato tensor
    document_embeddings = torch.tensor(document_embeddings, dtype=torch.float32)
    
    # Calcular similaridade de cosseno entre a consulta e os documentos
    similarities = F.cosine_similarity(user_embedding, document_embeddings).flatten()
    
    # Preparar os resultados com título, resumo e relevância
    results = []
    for i, doc in enumerate(documents):
        result = {
            'title': doc['title'],
            'abstract': ' '.join(doc['abstract'].split()[:500]),  # Limitar às primeiras 500 palavras
            'relevance': similarities[i].item()  # Converter tensor para valor numérico
        }
        if similarities[i] > 0.85:
            results.append(result)
    
    # Ordenar os resultados pela relevância em ordem decrescente
    sorted_results = sorted(results, key=lambda x: x['relevance'], reverse=True)
    
    # Retornar os 10 resultados mais relevantes
    return sorted_results[:10]
