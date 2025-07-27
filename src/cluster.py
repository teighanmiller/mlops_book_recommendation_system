"""
File for clustering algorithms
"""
import tempfile
from faiss import Kmeans
import boto3
import faiss
import mlflow.pyfunc
from src.faiss_model import FaissModel
from src.utility import write_data, get_data, isolate_embeddings


def faiss_cluster(input_path, output_path, params):
    """
    Performs FAISS kmeans clustering on data
    """

    with mlflow.start_run(run_name=f"best_faiss_kmeans_model_k_{params['k']}"):
        data = get_data(input_path)
        embeddings = isolate_embeddings(data)

        kmeans = Kmeans(**params)
        kmeans.train(embeddings)

        _, labels = kmeans.index.search(embeddings, k=1)

        data["cluster"] = labels

        write_data(data, output_path)

        with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as tmp_f:
            faiss.write_index(kmeans.index, tmp_f.name)
            index_path = tmp_f.name

        s3 = boto3.client("s3")
        s3_bucket, s3_key = output_path.replace("s3://", "").split("/", 1)
        faiss_key = s3_key.replace(".parquet", ".index")
        s3.upload_file(index_path, s3_bucket, faiss_key)
        s3_index_uri = f"s3://{s3_bucket}/{faiss_key}"

        mlflow.log_params(params)
        mlflow.pyfunc.log_model(
            name="best_model",
            python_model=FaissModel(),
            artifacts={"faiss_index": s3_index_uri},
        )

        result = mlflow.register_model(
            model_uri="runs:/" + mlflow.active_run().info.run_id + "/best_model",
            name="best-faiss-kmeans-model",
        )

        print(f"Registered model version: {result.version}")
