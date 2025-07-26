"""
This is the code file for finding the best cluster parameters.
"""

from datetime import date
import numpy as np
import mlflow
from faiss import Kmeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from src.utility import get_data, parse_io_args

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("faiss_kmeans_optimization")


def combined_score(silhouet_score, davies_score, calinski_score):
    """
    Creates a combined silhouette score, davies bouldin score, and carlinski harabasz score
    """
    return (silhouet_score * 100) - davies_score + (calinski_score / 1000)


### 146 clusters is optimal
def cluster_kmeans(data: np.ndarray, in_params: dict):
    """
    Performs clustering using KMeans
    """
    with mlflow.start_run():
        mlflow.set_tag("run_group_id", date.today().isoformat())

        params = {
            "d": data.shape[1],
            "k": in_params["n_clusters"],
            "niter": in_params["niter"],
            "verbose": True,
            "spherical": True,
        }

        kmeans = Kmeans(**params)
        kmeans.train(data)
        _, labels = kmeans.index.search(data, k=1)
        labels = labels.ravel()
        inertia = kmeans.obj[-1]

        sil_score = silhouette_score(data, labels)
        dav_score = davies_bouldin_score(data, labels)
        cal_score = calinski_harabasz_score(data, labels)

        metrics = {
            "silhouette_score": sil_score,
            "davies_bouldin_score": dav_score,
            "valinski_harabasz_score": cal_score,
            "overall_score": combined_score(sil_score, dav_score, cal_score),
            "inertia": inertia,
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)


def optimize(input_path: str, in_params: dict):
    """
    Loads data and performs clustering over a range of k values.
    """
    data = get_data(input_path)

    embeddings = data.drop(
        columns=["title", "author", "description", "genres"]
    ).to_numpy()

    for k in range(in_params["min_clusters"], in_params["max_clusters"] + 1, 2):
        in_params["n_clusters"] = k
        cluster_kmeans(embeddings, in_params)


if __name__ == "__main__":
    args = parse_io_args()

    if not args.input_path:
        raise ValueError("You must provide --input_path")

    arg_params = {
        "min_clusters": args.min_clusters,
        "max_clusters": args.max_clusters,
        "niter": args.niter,
    }

    optimize(args.input_path, arg_params)
