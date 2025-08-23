"""
Finds 10 recommendations using chosen book and clustering data
"""

from sklearn.metrics.pairwise import cosine_similarity
from src.utility import get_data, parse_io_args, isolate_embeddings


def find_recommendations(index: int, input_path: str):
    """
    Finds recommendations based on data passed
    """

    data = get_data(input_path)

    book = data.iloc[index]

    cluster = book["cluster"]

    book_embedding = book.drop(
        ["title", "genres", "author", "description", "cluster"]
    ).to_numpy()

    clustered_books = data[data["cluster"] == cluster].drop(columns=["cluster"])

    embeddings = isolate_embeddings(clustered_books)

    cosine_scores = []
    for embedding in embeddings:
        cosine_scores.append(
            cosine_similarity(book_embedding.reshape(1, -1), embedding.reshape(1, -1))
        )

    index_list = list(enumerate(cosine_scores))

    best_list = sorted(index_list, reverse=True)[1:12]

    print(best_list)


if __name__ == "__main__":
    args = parse_io_args()

    if not args.input_path:
        raise ValueError("You must provide --input_path")

    find_recommendations(9, args.input_path)
