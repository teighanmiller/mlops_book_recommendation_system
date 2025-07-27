"""
Workflow for searching database for specific books
"""
from rank_bm25 import BM25Okapi
from src.utility import get_data, parse_io_args


def run_search(input_file: str, text: str):
    """
    Gets top 10 results from search
    """
    data_df = get_data(input_file)
    data = data_df["corpus"].to_list()
    bm25 = BM25Okapi(data)
    query = text.split(" ")

    books = bm25.get_top_n(query, data, n=10)
    df_search = data_df[data_df["corpus"].isin(books)]
    return df_search.index


if __name__ == "__main__":
    args = parse_io_args()

    run_search(args.input_path, "Harry")
