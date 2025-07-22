from airflow.sdk import DAG
from datetime import timedelta, datetime
from airflow.providers.standard.operators.bash import BashOperator

with DAG(
    'book-recommendation-system-workflow',

    default_args={
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    },

    description="DAG workflow for cleaning, feature engineering, embeddings, and clustering data for book recommendation system.",
    schedule=timedelta(days=1),
    start_date=datetime(2025, 7, 21),
    catchup=False,
    tags=["book-recommendation"]

) as dag:
    clean = BashOperator(
        task_id="data_cleaning",
        bash_command="src/clean_data.py"
    )

    features = BashOperator(
        task_id="feature_engineering",
        bash_command="src/features.py"
    )

    embed = BashOperator(
        task_id="create_embeddings",
        bash_command="src/embeddings.py"
    )

    cluster = BashOperator(
        task_id="kmeans_clustering",
        bash_commands="src/kmeans_clustering.py"
    )

    clean >> features >> embed >> cluster