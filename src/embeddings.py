import os
import pandas as pd
from openai import AzureOpenAI
from requests_ratelimiter import LimiterSession
from token_limiter import TokenLimiter
from dotenv import load_dotenv
from token_utils import count_tokens
from clean_data import read_data
from features import create_features
from tqdm import tqdm

def _get_embeddings(texts: str | list, limiter: TokenLimiter, session: LimiterSession, model="text-embedding-3-small"):

    if type(texts) != str and type(texts) != list:
        raise ValueError("Must pass a string or a list to create embeddings")

    load_dotenv("/home/ubuntu/notebooks/mlops_book_recommendation_system/.env")

    with session:
        client = AzureOpenAI(
            api_key = os.getenv("AZURE_OPENAI_API_KEY"),
            api_version = "2024-10-21",
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        token_count = count_tokens(texts)
        limiter.wait_for_slot(token_count)

        response = client.embeddings.create(
            model = model,
            input = texts
        )

        return response.data

def get_embeddings(data: str | list, step=10) -> list:
    embeddings = [] 

    token_limiter = TokenLimiter()
    limiter_session = LimiterSession(per_minute=700) # Change to variable

    if type(data) == list:
        for i in tqdm(range(0, len(data), step)):
            batch = _get_embeddings(data[i:i+step], token_limiter, limiter_session)
            embeddings += [item.embedding for item in batch]
    elif type(data) == str:
        return _get_embeddings(data, token_limiter, limiter_session).data[0].embedding
    else:
        raise ValueError(f"Data of type {type(data)} is not supported. Please use string or list.")

    return embeddings


def run_embedding_process(categorical: pd.DataFrame, numerical: pd.DataFrame):
    print("Getting embeddings......")
    categorical_list = categorical.values.tolist()

    embeddings = get_embeddings(categorical_list)
    emb_len = len(embeddings[-1])

    categorical_df = pd.DataFrame(embeddings)
    numerical = pd.DataFrame(numerical, columns=[f"{emb_len+1}", f"{emb_len+2}"])

    embeddings_df = pd.concat([categorical_df, numerical], axis=1)
    embeddings_df.to_csv("data/open_ai_embeddings.csv", index=False)
    print("Done creating embeddings.")

# Works for testing 
if __name__=="__main__":
    filepath = r"/home/ubuntu/notebooks/mlops_book_recommendation_system/data/books_1.Best_Books_Ever.csv"
    clean_df = read_data(filepath)
    categorical, numerical = create_features(clean_df)
    run_embedding_process(categorical, numerical)


