"""
Workflow for searching database for specific books
"""
from rank_bm25 import BM25Okapi
from utility import get_data, parse_io_args


def run_search(input_file: str, text: str):
    """
    Gets top 10 results from search
    """
    data_df = get_data(input_file)
    data = data_df["corpus"].to_list()
    bm25 = BM25Okapi(data)
    query = text.split(" ")
    scores = bm25.get_scores(query)

    books = bm25.get_top_n(query, data, n=10)
    df_search = data_df[data_df["Text"].isin(books)]
    print(df_search.head())
    print(scores)


if __name__ == "__main__":
    args = parse_io_args()

    run_search(args.input_path, "Harry")
