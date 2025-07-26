"""
Files for model selection and upload
"""

import mlflow
from src.cluster import faiss_cluster
from src.utility import parse_io_args, check_io_args

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("faiss_kmeans_optimization")


def select_model(input_path: str, output_path: str):
    """
    Seaches mlflow logs for best cluster model and uploads it.
    """
    best_run = mlflow.search_runs(
        experiment_names=["faiss_kmeans_optimization"],
        order_by=["metrics.silhouette_score DESC"],
        max_results=1,
    )

    best_params = best_run.iloc[0].filter(like="params.").to_dict()
    best_params = {k.replace("params.", ""): int(v) for k, v in best_params.items()}

    faiss_cluster(input_path, output_path, **best_params)


if __name__ == "__main__":
    args = parse_io_args()

    check_io_args(args)

    select_model(args.input_path, args.output_path)
