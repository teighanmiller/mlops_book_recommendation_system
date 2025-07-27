"""
Files for model selection and upload
"""

from datetime import date, timedelta
import mlflow
import numpy as np
import pandas as pd
from kneed import KneeLocator
from src.cluster import faiss_cluster
from src.utility import parse_io_args, check_io_args

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("faiss_kmeans_optimization")


def select_model(input_path: str, output_path: str):
    """
    Seaches mlflow logs for best cluster model and uploads it.
    """

    models = mlflow.search_runs(
        experiment_names=["faiss_kmeans_optimization"],
        output_format="pandas",
        max_results=200,
    )

    daily_models = pd.DataFrame(columns=models.columns)
    check_date = date.today()
    cut_date = date.today() - timedelta(days=20)
    while daily_models.empty and cut_date < check_date:
        daily_models = models[models["tags.run_group_id"] == check_date.isoformat()]
        check_date -= timedelta(days=1)

    if daily_models.empty:
        raise ValueError(f"Could not find models after {cut_date}")

    metrics = daily_models[["params.k", "metrics.inertia"]]

    k_list = sorted(np.array(metrics["params.k"].to_list(), dtype=int))
    inertias_list = sorted(np.array(metrics["metrics.inertia"].to_list(), dtype=int))

    kn = KneeLocator(k_list, inertias_list, curve="concave", direction="increasing")

    best_model = daily_models[pd.to_numeric(daily_models["params.k"]) == int(kn.knee)]

    if len(best_model) > 1:
        best_model = best_model.iloc[0]
    elif len(best_model) < 1:
        raise ValueError("Best model not found.")

    params = {
        "d": int(pd.to_numeric(best_model["params.d"]).iloc[0]),
        "k": int(pd.to_numeric(best_model["params.k"]).iloc[0]),
        "niter": int(pd.to_numeric(best_model["params.niter"]).iloc[0]),
        "verbose": bool(best_model["params.verbose"].iloc[0]),
        "spherical": bool(best_model["params.spherical"].iloc[0]),
    }

    faiss_cluster(input_path, output_path, params)


if __name__ == "__main__":
    args = parse_io_args()

    check_io_args(args)

    select_model(args.input_path, args.output_path)
