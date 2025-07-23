"""

This is the DAG workflow for data ingestion in the book recommendations system project.

"""
from datetime import timedelta, datetime
from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator

with DAG(
    'book-recommendation-system-workflow',

    default_args={
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    },

    description="DAG workflow for cleaning, feature engineering, embeddings, \
        and clustering data for book recommendation system.",
    schedule=timedelta(days=1),
    start_date=datetime(2025, 7, 21),
    catchup=False,
    tags=["book-recommendation"]

) as dag:
    clean = BashOperator(
        task_id="data_cleaning",
        bash_command=(
                "python /opt/airflow/src/clean_data.py "
                "--output_path s3://books-recommendation-storage/data \
                    /cleaned_{{ ds_nodash }}.parquet"
            )
    )

    features = BashOperator(
        task_id="feature_engineering",
        bash_command=(
            "python /opt/airflow/src/features.py "
            "--input_path s3://books-recommendation-storage/data/cleaned_{{ ds_nodash }}.parquet "
            "--output_path s3://books-recommendation-storage/data/feature_{{ ds_nodash }}.parquet"
        )
    )

    embed = BashOperator(
        task_id="create_embeddings",
        bash_command=(
            "python /opt/airflow/src/embeddings.py "
            "--input_path s3://books-recommendation-storage/data/feature_{{ ds_nodash }}.parquet "
            "--output_path s3://books-recommendation-storage/data/ \
                embeddings_{{ ds_nodash }}.parquet"
        )
    )

    clean >> features >> embed # pylint: disable=pointless-statement
