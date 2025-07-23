import mlflow
import argparse
from utility import upload_to_s3

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("book_clustering_experiment")

def select_model(output_path: str):
    top_models = mlflow.search_logged_models(
        experimet_ids=["1", "2"],
        filter_string = "metrics.silhouette_score > 0.70",
        order_by=[{"field_name": "metrics.silhouette_score", "ascending": False}],
        max_results = 5,
    )

    best_model = mlflow.search_logged_models(
        experiment_ids=["1"],
        filter_string = "metrics.silhouette_score > 0.70",
        max_results = 1,
        order_by=[{"field_name": "metrics.silhouette_score", "ascending": False}],
        output_format="list"
    )[0]

    loaded_model = mlflow.pyfunc.load_model(f"models:/{best_model.model_id}")

    if output_path.startswith("s3://"):
        _, _, bucket, *key_parts = output_path.split("/")
        s3_key = "/".join(key_parts)
        ## SHOULD CHANGE THIS TO ACCEPT MODEL NOT PARQUET
        upload_to_s3(loaded_model, bucket, s3_key)
    else:
        raise ValueError("Output path must start with s3://")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", "-o", type=str, required=True, help="s3://bucket-name/path/to/output.csv")
    args = parser.parse_args()
    select_model(args.output_path)