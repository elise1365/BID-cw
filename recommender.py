import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import ast
from sklearn.feature_extraction.text import TfidfVectorizer

# Remove unreleased films from the dataset
def cleanDataset(df):
    df = df[df['status'] != 'Unreleased']
    return df

def reduceToNames(obj):
    lst = []
    if obj:
        for i in ast.literal_eval(obj):
            lst.append(i['name'])
    return lst

def contentBasedRecommender(similarityMatrix, itemIndex):
    # Gets the column of distances for the movie being asked about
    column = similarityMatrix[itemIndex]
    # Sort column so that top 6 highest scores are stored
    lst = column.argsort()[-6:]
    # Remove the first film from the list, this is the movie we are asking for recomendations from
    lst = [i for i in lst if i != itemIndex]
    # List of movie titles
    labels = movies['title'].tolist()
    # Convert list of similar scores into their corresponding movie titles
    similar_movies = []
    # Filter lst by top 1000 movies ordered by revenue
    top_1000_movies = [i for i in lst if i < 1000]
    for i in top_1000_movies:
        similar_movies.append(labels[i])
        if len(similar_movies) >= 5:
            break
    # If less than 5 movies are found within top 1000, add from the overall list
    if len(similar_movies) < 5:
        remaining = 5 - len(similar_movies)
        for i in lst:
            if i >= 1000:
                similar_movies.append(labels[i])
                remaining -= 1
                if remaining == 0:
                    break
    return "If you liked '" + labels[itemIndex] + "' then you'd probably like: '" + "', '".join(similar_movies) + "'."

# Open datasets
movies = pd.read_csv(r"C:\Users\pokem\OneDrive\Documents\tmdb_5000_movies.csv")
credits = pd.read_csv(r"C:\Users\pokem\OneDrive\Documents\tmdb_5000_credits.csv")

movies = cleanDataset(movies)

movies['cast'] = credits['cast']

# Sort movies by revenue
movies = movies.sort_values('revenue', ascending=False)

# Remove the ids from genres and keywords
movies['genres'] = movies['genres'].apply(reduceToNames)
movies['keywords'] = movies['keywords'].apply(reduceToNames)
movies['cast'] = movies['cast'].apply(reduceToNames)

movies['genres'] = movies['genres'].apply(lambda x: ' '.join(x))
movies['keywords'] = movies['keywords'].apply(lambda x: ' '.join(x))
movies['cast'] = movies['cast'].apply(lambda x: ' '.join(x))

movies['all_attributes'] = movies['genres'] + movies['keywords'] + movies['cast']

# Tfid vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(movies['all_attributes'])

# Create similarity matrix
similarityMatrix = 1 - pairwise_distances(tfidf_matrix, metric='cosine')
similarityMatrix = np.transpose(similarityMatrix)

print(contentBasedRecommender(similarityMatrix, 0))
print(contentBasedRecommender(similarityMatrix, 15))

