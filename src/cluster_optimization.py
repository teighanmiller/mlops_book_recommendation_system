"""
This is the code file for finding the best cluster parameters.
"""

import numpy as np
import mlflow
from faiss import Kmeans
from sklearn.metrics import silhouette_score

from src.utility import get_data, parse_io_args

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("faiss_kmeans_optimization")


def cluster_kmeans(data: np.ndarray, in_params: dict):
    """
    Performs clustering using KMeans
    """
    with mlflow.start_run():

        params = {
            "d": data.shape[1],
            "n_centroids": in_params["n_clusters"],
            "niter": in_params["niter"],
            "random_state": in_params["random_state"],
            "verbose": True,
            "spherical": True,
        }

        kmeans = Kmeans(**params)
        kmeans.train(data)
        _, labels = kmeans.index.search(data, k=1)
        sil_score = silhouette_score(X=data, labels=labels)

        mlflow.log_params(params)
        mlflow.log_metric("silhouette_score", sil_score)


def optimize(input_path: str, in_params: dict):
    """
    Loads data and performs clustering over a range of k values.
    """
    data = get_data(input_path).to_numpy()

    for k in range(in_params["min_clusters"], in_params["max_clusters"] + 1, 2):
        in_params["n_clusters"] = k
        cluster_kmeans(data, in_params)


if __name__ == "__main__":
    args = parse_io_args()

    if not args.input_path:
        raise ValueError("You must provide --input_path")

    arg_params = {
        "min_clusters": args.min_clusters,
        "max_clusters": args.max_clusters,
        "random_state": args.random_state,
        "niter": args.niter,
    }

    optimize(args.input_path, arg_params)
