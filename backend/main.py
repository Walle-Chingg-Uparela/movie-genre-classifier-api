from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
import pandas as pd
import re
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from scipy.sparse import hstack, csr_matrix


app = FastAPI(
    title="Movie Genre Classifier API",
    description="Predict movie genres using title and optional plot",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

nltk.download('stopwords')
nltk.download('wordnet')


model = joblib.load('model.pkl')

vect_plot = joblib.load('vectorizer_plot.pkl')

vect_title = joblib.load('vectorizer_title.pkl')

scaler = joblib.load('scaler.pkl')

mlb = joblib.load('mlb.pkl')



app = FastAPI(
    title="Movie Genre Classifier API",
    description="Predict movie genres using title and optional plot",
    version="2.0.0"
)



lemmatizer = WordNetLemmatizer()

extra_stop_words = [
    'life', 'one', 'find', 'get', 'new', 'man', 'year',
    'time', 'two', 'takes', 'story', 'film', 'come',
    'way', 'go', 'make', 'see', 'like', 'know',
    'people', 'would', 'could', 'also', 'first',
    'well', 'even', 'back', 'much', 'must',
    'say', 'tell'
]

my_stop_words = set(stopwords.words('english')).union(extra_stop_words)


def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r'[^a-z\s]', '', text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in my_stop_words and len(word) > 2
    ]

    return ' '.join(words)



def create_features(title, plot, year):

    title_clean = clean_text(title)

    plot_clean = clean_text(plot)

    features = {
        "title_clean": title_clean,
        "plot_clean": plot_clean,
        "plot_len": len(plot_clean),
        "plot_words": len(plot_clean.split()),
        "title_len": len(title_clean),
        "title_words": len(title_clean.split()),
        "year_norm": (year - 1900) / (2025 - 1900)
    }

    return features



class MovieRequest(BaseModel):

    title: str

    plot: str = ""

    year: int = 2024



@app.get("/")

def home():

    return {
        "message": "Movie Genre Classifier API is running successfully"
    }



@app.get("/health")

def health():

    return {
        "status": "ok"
    }



@app.post("/predict")

def predict(movie: MovieRequest):


    features = create_features(
        title=movie.title,
        plot=movie.plot,
        year=movie.year
    )

    

    X_plot = vect_plot.transform([features['plot_clean']])

    X_title = vect_title.transform([features['title_clean']])




    numeric_features = np.array([[
        features['plot_len'],
        features['plot_words'],
        features['title_len'],
        features['title_words'],
        features['year_norm']
    ]])

    X_num = scaler.transform(numeric_features)


    X_full = hstack([
        X_plot,
        X_title,
        csr_matrix(X_num)
    ])

 

    probabilities = model.predict_proba(X_full)[0]

   

    predictions = {}

    for genre, probability in zip(mlb.classes_, probabilities):

        predictions[genre] = round(float(probability), 4)



    predictions = dict(
        sorted(
            predictions.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )



    top_genres = list(predictions.items())[:5]



    return {
        "movie_title": movie.title,
        "year": movie.year,
        "plot_provided": bool(movie.plot.strip()),
        "top_predictions": top_genres,
        "all_predictions": predictions
    }
