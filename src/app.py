"""
Backend for book recommendation application
"""
from flask import Flask, request, render_template, session
from flask_session import Session

app = Flask(__name__)
app.config["SECRET_KEY"] = "your_secret_key_here"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Mock search function
def mock_search(query):
    """
    search function
    """
    if not query.strip():
        return []
    return [f"{query} Result {i+1}" for i in range(10)]


@app.route("/", methods=["GET", "POST"])
def search():
    """
    Search webpage
    """
    results = []
    if request.method == "POST":
        query = request.form.get("query", "")
        results = mock_search(query)
        session["last_query"] = query
        session["last_results"] = results
    return render_template("search.html", results=enumerate(results))


@app.route("/results/<int:item_id>")
def result_detail(item_id):
    """
    Renders search results
    """
    return render_template("search_result.html", item_id=item_id)


@app.route("/results/<int:parent_id>/related/<int:related_id>")
def related_search(parent_id, related_id):
    """
    Renders clustering results.
    """
    base_query = session.get("last_query", "Item")
    related_query = f"{base_query} Detail {parent_id} - Related {related_id}"
    results = mock_search(related_query)
    return render_template(
        "recommendations.html",
        parent_id=parent_id,
        related_id=related_id,
        results=results,
    )


if __name__ == "__main__":
    app.run(debug=True)
