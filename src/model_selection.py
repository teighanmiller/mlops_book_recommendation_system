"""
Files for model selection and upload
"""
import mlflow
from src.utility import write_data, parse_io_args

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("book_clustering_experiment")

def select_model(output_path: str):
    """
    Seaches mlflow logs for best cluster model and uploads it.
    """
    best_model = mlflow.search_logged_models(
        experiment_ids=["1"],
        filter_string = "metrics.silhouette_score > 0.70",
        max_results = 1,
        order_by=[{"field_name": "metrics.silhouette_score", "ascending": False}],
        output_format="list"
    )[0]

    loaded_model = mlflow.pyfunc.load_model(f"models:/{best_model.model_id}")

    write_data(loaded_model, output_path)
    
if __name__=="__main__":
    args = parse_io_args()

    if not args.output_path:
        raise ValueError("You must provide --output_path")

    select_model(args.output_path)
