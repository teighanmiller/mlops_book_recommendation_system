"""
This is the script to create embeddings using local embedding model.
"""

import pandas as pd
from sentence_transformers import SentenceTransformer

from src.utility import check_io_args, get_data, parse_io_args, write_data


def create_embeddings(data: str | list) -> list:
    """
    Encodes data using mxbai large model
    """
    model = SentenceTransformer("mixedbread-ai/mxbai-embed-xsmall-v1")
    embeddings = model.encode(data)
    return embeddings


def get_embeddings(df: pd.DataFrame) -> None:
    """
    Seperates numeric and text data. Scales numeric data and embeds text data.
    """
    categorical = df["categorical"]
    numerical = df[["scaled_percent", "scaled_ratings"]]

    print("Getting embeddings......")

    if len(categorical) != len(numerical):
        raise ValueError("Numerical and Categorical datasets must be the same size.")

    categorical_list = categorical.values.tolist()

    embeddings = create_embeddings(categorical_list)
    emb_len = len(embeddings[-1])

    categorical_df = pd.DataFrame(embeddings)
    numerical.columns = [f"{emb_len + 1}", f"{emb_len + 2}"]

    embeddings_df = pd.concat([categorical_df, numerical], axis=1)

    return embeddings_df


def run_embedding_process(input_path: str, output_path: str) -> None:
    """
    Runs the full embedding process including loading and writing data.
    """
    df = get_data(input_path)

    # No need to pass all this data.
    embeddings = get_embeddings(df[["categorical", "scaled_percent", "scaled_ratings"]])
    full_df = pd.concat(
        [df[["title", "author", "description", "genres"]], embeddings], axis=1
    )

    write_data(full_df, output_path)
    print("Done creating embeddings.")


if __name__ == "__main__":
    args = parse_io_args()

    check_io_args(args)

    run_embedding_process(args.input_path, args.output_path)
