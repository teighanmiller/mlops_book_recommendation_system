"""
Utility functions for the mlops book recommendation system
"""

import argparse
from io import BytesIO

import boto3
import botocore.exceptions
import pandas as pd


def get_data(input_path: str) -> pd.DataFrame:
    """
    Parse input_path for retrieval from s3 bucket
    """
    if input_path.startswith("s3://"):
        _, _, bucket, *key_parts = input_path.split("/")
        s3_key = "/".join(key_parts)
        df = get_from_s3(bucket, s3_key)
    else:
        raise ValueError("Input path must start with s3://")
    return df


def write_data(data, output_path: str) -> None:
    """
    Parse output_path for uploading data to s3 bucket
    """
    if output_path.startswith("s3://"):
        _, _, bucket, *key_parts = output_path.split("/")
        s3_key = "/".join(key_parts)
        upload_to_s3(data, bucket, s3_key)
    else:
        raise ValueError("Output path must start with s3://")


def upload_to_s3(df: pd.DataFrame, bucket_name: str, s3_key: str) -> None:
    """
    Function to upload dataframe as a parquet to s3 bucket.
    """
    try:
        s3 = boto3.client("s3")
        buffer = BytesIO()
        df.to_parquet(buffer, index=False)
        s3.put_object(Bucket=bucket_name, Key=s3_key, Body=buffer.getvalue())
    except (
        botocore.exceptions.BotoCoreError,
        botocore.exceptions.ClientError,
        ValueError,
    ) as e:
        print(f"An exception has occured during upload: {e}")


def get_from_s3(bucket_name: str, file_key: str) -> pd.DataFrame:
    """
    Function to download parquet from s3 bucket
    """
    try:
        s3 = boto3.client("s3")
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        status_code = response["ResponseMetadata"]["HTTPStatusCode"]
        if status_code == 200:
            parquet_contents = BytesIO(response["Body"].read())
            data = pd.read_parquet(parquet_contents)
        else:
            print(f"Error getting object: HTTP status code {status_code}")
            return None

        return data
    except (
        botocore.exceptions.BotoCoreError,
        botocore.exceptions.ClientError,
        ValueError,
    ) as e:
        print(f"An error ocurred: {e}")
        return None


def parse_io_args():
    """
    Argument parser for input and output files
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", "-i", type=str, help="s3://bucket-name/path/to/output.csv"
    )
    parser.add_argument(
        "--output_path", "-o", type=str, help="s3://bucket-name/path/to/output.csv"
    )
    parser.add_argument(
        "--min_clusters",
        "-m",
        type=int,
        default=2,
        help="minimum number of clusters to check",
    )
    parser.add_argument(
        "--max_clusters",
        "-x",
        type=int,
        default=50,
        help="maximum number of clusters to check",
    )
    parser.add_argument(
        "--niter",
        "-n",
        type=int,
        default=20,
        help="iterations run for kmeans clustering model",
    )
    return parser.parse_args()


def check_io_args(args):
    """
    Checks if input_path and output_path are present in args
    """
    if not args.input_path:
        raise ValueError("You must provide --input_path")

    if not args.output_path:
        raise ValueError("You must provide --ouput_path")
