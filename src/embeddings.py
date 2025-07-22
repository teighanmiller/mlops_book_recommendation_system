import pandas as pd
import argparse
from sentence_transformers import SentenceTransformer
from utility import get_from_s3, upload_to_s3

def get_embeddings(data: str | list) -> list:
    model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
    embeddings = model.encode(data)
    return embeddings

def run_embedding_process(input_path: str, output_path: str):
    if input_path.startswith("s3://"):
        _, bucket, *key_parts = input_path.split("/")
        s3_key = "/".join(key_parts)
        df = get_from_s3(bucket, s3_key)
    else:
        raise ValueError("Input path must start with s3://")

    categorical = df['categorical']
    numerical = df['scaled_percent', 'scaled_ratings']

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
        _, bucket, *key_parts = output_path.split("/")
        s3_key = "/".join(key_parts)
        upload_to_s3(embeddings_df, bucket, s3_key)
    else:
        raise ValueError("Output path must start with s3://")

    print("Done creating embeddings.")

# Works for testing 
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", type=str, required=True, help="s3://bucket-name/path/to/output.csv")
    parser.add_argument("--output_path", "-o", type=str, required=True, help="s3://bucket-name/path/to/output.csv")
    args = parser.parse_args()
    run_embedding_process(args.input_path, args.output_path)


