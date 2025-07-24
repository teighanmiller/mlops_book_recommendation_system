"""
This is the code file for finding the best cluster parameters.
"""
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from src.utility import get_from_s3, parse_io_args

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("book_clustering_experiment")

def cluster_kmeans(data: np.ndarray, in_params: dict):
    """
    Performs clustering using KMeans
    """
    with mlflow.start_run():

        params = {
            "n_clusters": in_params['n_clusters'],
            'random_state': in_params['random_state']
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

def optimize(input_path: str, in_params: dict):
    """
    Loads data and performs clustering over a range of k values.
    """
    if input_path.startswith("s3://"):
        _, _, bucket, *key_parts = input_path.split("/")
        s3_key = "/".join(key_parts)
        data = get_from_s3(bucket, s3_key)
    else:
        raise ValueError("Output path must start with s3://")

    for _ in range(in_params['min_clusters'], in_params['max_clusters'] + 1, 2):
        cluster_kmeans(data, in_params)

if __name__=="__main__":
    args = parse_io_args()

    if not args.input_path:
        raise ValueError("You must provide --input_path")
    
    if not args.output_path:
        raise ValueError("You must provide --ouput_path")

    # Need to add conditions if parameters are empty
    arg_params = {
        "output_path": args.output_path,
        "min_clusters": args.min_clusters,
        "max_clusters": args.max_clusters,
        "random_state": args.random_state
    }

    optimize(args.input_path, arg_params)
