from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

papers = pd.DataFrame({
    "title": [
        "Deep Learning for NLP",
        "Machine Learning Basics",
        "Computer Vision using CNN",
        "Transformers in AI",
        "Recommendation Systems",
        "Neural Networks Explained"
    ],
    "abstract": [
        "Natural language processing using deep learning techniques",
        "Introduction to machine learning algorithms and models",
        "Image classification and object detection using convolutional neural networks",
        "Transformer architecture for artificial intelligence applications",
        "Content based and collaborative filtering recommendation systems",
        "Understanding neural networks and deep learning fundamentals"
    ]
})

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(papers['abstract'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_papers(title):
    idx = papers[papers['title'] == title].index[0]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:4]

    recommendations = []
    for i in similarity_scores:
        recommendations.append(papers.iloc[i[0]]['title'])

    return recommendations

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Research Paper Recommendation System</title>
</head>
<body style="font-family: Arial; padding: 40px;">
    <h1>Research Paper Recommendation System</h1>

    <form method="POST">
        <select name="paper">
            {% for paper in papers %}
            <option value="{{paper}}">{{paper}}</option>
            {% endfor %}
        </select>

        <button type="submit">Recommend</button>
    </form>

    {% if recommendations %}
        <h2>Recommended Papers:</h2>
        <ul>
        {% for rec in recommendations %}
            <li>{{rec}}</li>
        {% endfor %}
        </ul>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []

    if request.method == "POST":
        paper = request.form["paper"]
        recommendations = recommend_papers(paper)

    return render_template_string(
        HTML,
        papers=papers["title"].tolist(),
        recommendations=recommendations
    )

if __name__ == "__main__":
    app.run(debug=True)