from flask import Flask, render_template, request
import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download movie review dataset from nltk
nltk.download("movie_reviews")

# Prepare the data
documents = [
    (" ".join(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]

# Prepare the DataFrame
import pandas as pd
df = pd.DataFrame(documents, columns=["review", "sentiment"])

# Vectorize the text data
vectorizer = CountVectorizer(max_features=2000)
X = vectorizer.fit_transform(df["review"])
y = df["sentiment"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Define a prediction function
def predict_sentiment(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return prediction[0]

# Create the Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = ""
    review = ""
    if request.method == "POST":
        review = request.form["review"]
        sentiment = predict_sentiment(review)
    return render_template("index.html", sentiment=sentiment, review=review)

if __name__ == "__main__":
    app.run(debug=True)