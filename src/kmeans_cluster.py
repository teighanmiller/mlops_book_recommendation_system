import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import mlflow
from mlflow.models import infer_signature

EMBEDDING_PATH = "/home/ubuntu/notebooks/mlops_book_recommendation_system/data/open_ai_embeddings.csv"

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("book_clustering_experiment")

def cluster_kmeans(data: np.ndarray, n_clusters, random_state=42):
    with mlflow.start_run():

        params = {
            "n_clusters": n_clusters,
            'random_state': random_state
        }

        kmeans = KMeans(**params).fit(data)
        labels = kmeans.predict(data)
        sil_score = silhouette_score(X=data, labels=labels)

        mlflow.log_params(params)
        mlflow.log_metric("silhouette_score", sil_score)

        signature = infer_signature(data, kmeans.predict(data))

        model_info = mlflow.sklearn.log_model(
            sk_model=KMeans,
            name='book_clustering',
            signature=signature,
            input_example=data,
        )

        mlflow.set_logged_model_tags(
            model_info.model_id,
            {"Training Info: Test Kmeans clustering model for book recommedation system."}
        )

def run_clustering(filepath: str, min_clusters: int, max_clusters: int):
    embedding_df = pd.read_csv(filepath)
    for k in range(min_clusters, max_clusters + 1, 2):
        cluster_kmeans(embedding_df, k)

if __name__=="__main__":
    run_clustering(EMBEDDING_PATH, 2, 50)