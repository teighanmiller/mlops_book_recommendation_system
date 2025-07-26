"""
File for clustering algorithms
"""
from faiss import Kmeans
import mlflow.pyfunc
from faiss_model import FaissModel
from utility import write_data, get_data


def faiss_cluster(input_path, output_path, params):
    """
    Performs FAISS kmeans clustering on data
    """
    data = get_data(input_path).to_numpy()

    kmeans = Kmeans(**params)
    kmeans.train(data)

    write_data(kmeans.index, output_path)

    with mlflow.start_run(
        run_name=f"best_faiss_kmeans_model_k_{params['n_centroids']}"
    ):
        mlflow.log_params(params)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=FaissModel(),
            artifacts={"faiss_index": output_path},
        )

        result = mlflow.register_model(
            model_uri="runs:/" + mlflow.active_run().info.run_id + "/model",
            name="best-faiss-kmeans-model",
        )

        print(f"Registered model version: {result.version}")
