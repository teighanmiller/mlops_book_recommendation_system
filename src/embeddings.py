"""
This is the script to create embeddings using local embedding model.
"""
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.utility import get_from_s3, upload_to_s3, parse_io_args

def get_embeddings(data: str | list) -> list:
    """
    Encodes data using mxbai large model
    """
    model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
    embeddings = model.encode(data)
    return embeddings

def run_embedding_process(input_path: str, output_path: str) -> None:
    """
    Runs the full embedding process including loading and writing data
    """
    if input_path.startswith("s3://"):
        _, _, bucket, *key_parts = input_path.split("/")
        s3_key = "/".join(key_parts)
        df = get_from_s3(bucket, s3_key)
    else:
        raise ValueError("Input path must start with s3://")

    categorical = df['categorical']
    numerical = df[['scaled_percent', 'scaled_ratings']]

    print("Getting embeddings......")

    if len(categorical) != len(numerical):
        raise ValueError("Numerical and Categorical datasets must be the same size.")

    categorical_list = categorical.values.tolist()

    embeddings = get_embeddings(categorical_list)
    emb_len = len(embeddings[-1])

    categorical_df = pd.DataFrame(embeddings)
    numerical.columns = [f"{emb_len + 1}", f"{emb_len + 2}"]

    embeddings_df = pd.concat([categorical_df, numerical], axis=1)

    if output_path.startswith("s3://"):
        _, _, bucket, *key_parts = output_path.split("/")
        s3_key = "/".join(key_parts)
        upload_to_s3(embeddings_df, bucket, s3_key)
    else:
        raise ValueError("Output path must start with s3://")

    print("Done creating embeddings.")

if __name__=="__main__":
    args = parse_io_args()

    if not args.input_path:
        raise ValueError("You must provide --input_path")
    
    if not args.output_path:
        raise ValueError("You must provide --ouput_path")

    run_embedding_process(args.input_path, args.output_path)
