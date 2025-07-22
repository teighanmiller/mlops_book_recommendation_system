import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from clean_data import read_data
from features import create_features
from tqdm import tqdm

def get_embeddings(data: str | list) -> list:
    model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
    embeddings = model.encode(data)
    return embeddings

def run_embedding_process(categorical: pd.DataFrame, numerical: pd.DataFrame):
    print("Getting embeddings......")

    if len(categorical) != len(numerical):
        raise ValueError("Numerical and Categorical datasets must be the same size.")

    categorical_list = categorical.values.tolist()

    embeddings = get_embeddings(categorical_list)
    emb_len = len(embeddings[-1])

    categorical_df = pd.DataFrame(embeddings)
    numerical = pd.DataFrame(numerical)
    numerical.columns = [f"{emb_len + 1}", f"{emb_len + 2}"]

    embeddings_df = pd.concat([categorical_df, numerical], axis=1)
    embeddings_df.to_csv("data/open_ai_embeddings.csv", index=False)
    print("Done creating embeddings.")

# Works for testing 
if __name__=="__main__":
    filepath = r"/home/ubuntu/notebooks/mlops_book_recommendation_system/data/books_1.Best_Books_Ever.csv"
    clean_df = read_data(filepath)
    categorical, numerical = create_features(clean_df)
    run_embedding_process(categorical, numerical)


