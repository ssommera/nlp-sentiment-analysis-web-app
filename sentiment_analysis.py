import nltk
import pandas as pd
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import movie_reviews


# Step 2: Data Preparation
nltk.download("movie_reviews")

# Load dataset
documents = [
    (" ".join(movie_reviews.words(fileid)), category, fileid)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]

# Convert to DataFrame, making sure to use the correct columns
df = pd.DataFrame(documents, columns=["review", "sentiment", "fileid"])

# Step 3: Model Training

# Convert text data to feature vectors
#vectorizer = CountVectorizer(max_features=2000)
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(df["review"])
y = df["sentiment"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes Classifier
#model = MultinomialNB()
model = MultinomialNB(alpha=0.5)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Step 4: Prediction

def predict_statement(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return prediction[0]

# Test the prediction function
print(predict_statement("I absolutely loved this movie! It was fantastic."))
print(predict_statement("It was a terrible film. I hated it"))
print(predict_statement("The movie was okay, nothing special"))

# Show movie titles (filenames) for a sample of the reviews in the DataFrame
print("\nSample Movie Reviews with Titles and Sentiments:")
for index, row in df.sample(5).iterrows():  # Show 5 random reviews with movie titles
    print(f"Sentiment: {row['sentiment']} | Review: {row['review'][:150]}...")  # First 150 chars of review


# The distribution of positive and negative reviews using a bar chart or pie chart.
import matplotlib.pyplot as plt

# Plot the sentiment distribution
sentiment_counts = df['sentiment'].value_counts()
sentiment_counts.plot(kind='bar', color=['red', 'green'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=0)
plt.show()


# Generate Word Clouds for Each Sentiment
from wordcloud import WordCloud

# Create a word cloud for positive reviews
pos_reviews = " ".join(df[df['sentiment'] == 'pos']['review'])
wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(pos_reviews)

# Create a word cloud for negative reviews
neg_reviews = " ".join(df[df['sentiment'] == 'neg']['review'])
wordcloud_neg = WordCloud(width=800, height=400, background_color='white').generate(neg_reviews)

# Display word clouds
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.title('Positive Sentiment Word Cloud')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.title('Negative Sentiment Word Cloud')
plt.axis('off')

plt.tight_layout()
plt.show()

