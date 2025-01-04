# Sentiment Analysis Web App

A web application that uses Natural Language Processing (NLP) to perform sentiment analysis on movie reviews.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Requirements](#requirements)
- [Setup](#setup)
- [Using the App](#using-the-app)

## Features

- Sentiment analysis of movie reviews (positive or negative)
- Web interface for inputting reviews and receiving predictions
- Utilizes NLTK's movie review dataset for training
- Implements a Multinomial Naive Bayes classifier
- Achieves approximately 80% accuracy on test data

## Technologies Used

- Python 3.7+
- Flask (Web framework)
- NLTK (Natural Language Toolkit)
- scikit-learn (Machine Learning library)
- Poetry (Dependency management)

## Requirements

- Python (>= 3.7)
- Poetry (for managing dependencies)
- A web browser

## Setup

1. **Clone this repository:**
git clone https://github.com/yourusername/sentiment-analysis-web-app.git
cd sentiment-analysis-web-app

2.**Install dependencies using Poetry:**
poetry install flask nltk scikit-learn pandas

3. **Run the app:**
poetry run python app.py


## Using the App
1. Open your browser and go to http://127.0.0.1:5000/.
2. Input a movie review in the provided text box and click "Analyze Sentiment."
3. The app will display whether the sentiment of the review is positive or negative.