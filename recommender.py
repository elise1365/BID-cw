# Datasets put into dataframe, like a table, easy to handle data this way
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
# ast is used when removing ids from attributes
import ast

def cleanDataset(df):
    # Remove any entries with null values
    for index, row in df.iterrows():
        if row.isnull().any():
            df = df.drop(index)
    
    # Remove unreleased films from the dataset
    df = df[df['status'] != 'Unreleased']
    return df

# Remove ids from attributes
def reduceToNames(column):
    lst = []
    if column:
        for i in ast.literal_eval(column):
            lst.append(i['name'])
    return lst

def contentBasedRecommender(similarityMatrix, itemIndex, movies):
    # Gets the row of values for the movie being asked about
    row = similarityMatrix[itemIndex]
    # Sort row so that top 6 highest scores are stored
    lst = row.argsort()[-6:]
    # Remove the first film from the list, this is the movie we are asking for recomendations for
    lst = [i for i in lst if i != itemIndex]
    
    # List of movie titles
    labels = movies['title'].tolist()
    
    # Convert list of similar scores into their corresponding movie titles
    similar_movies = []
    # Filter list by top 1000 movies ordered by revenue
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
    
    # Nicely formatted output, user friendly            
    output = "If you liked '" + labels[itemIndex] + "' then you'd probably like: '" + "', '".join(similar_movies) + "'."
    return output

# Prepare the dataset in order to make a similarity matrix later on
def datasetPreprocessing():
    # Open datasets as csvs
    movies = pd.read_csv(r"C:\Users\pokem\OneDrive\Documents\tmdb_5000_movies.csv")
    credits = pd.read_csv(r"C:\Users\pokem\OneDrive\Documents\tmdb_5000_credits.csv")

    # Add cast to movies, means we only have to handle 1 dataframe, makes life easier
    movies['cast'] = credits['cast']

    # clean the dataset
    movies = cleanDataset(movies)

    # Sort movies by revenue
    movies = movies.sort_values('revenue', ascending=False)

    # Remove the ids from chosen attributes
    movies['genres'] = movies['genres'].apply(reduceToNames)
    movies['keywords'] = movies['keywords'].apply(reduceToNames)
    movies['cast'] = movies['cast'].apply(reduceToNames)
    # Convert chosen attributes from lists to strings, tfidf vectorizer can't handle lists
    movies['genres'] = movies['genres'].apply(lambda x: ' '.join(x))
    movies['keywords'] = movies['keywords'].apply(lambda x: ' '.join(x))
    movies['cast'] = movies['cast'].apply(lambda x: ' '.join(x))

    # Combine all chosen attributes into 1 column
    movies['all_attributes'] = movies['genres'] + movies['keywords'] + movies['cast']
    
    return movies

def createSimilarityMatrix(movies):
    # Initialised tfidf vectorizer to a variable so its callable
    vectorizer = TfidfVectorizer()
    # Apply the vectorizer to a tfidf matrix, using the all_attributes column prepared earlier
    tfidf_matrix = vectorizer.fit_transform(movies['all_attributes'])

    # Create similarity matrix using the vecto
    similarityMatrix = 1 - pairwise_distances(tfidf_matrix, metric='cosine')
    
    return similarityMatrix

# Implement the recommender system so its user friendly and ready to use
def systemImplementation():
    movies = datasetPreprocessing()

    # Take users input
    movie = input('Please enter a movie: ')
    movieIndex = 0
    # Ensure the movie exists
    for index, row in movies.iterrows():
        if row['title'] == movie:
            movieIndex = index
            # Only create similarity matrix when movie index is found, avoids wasting time if movie doesn't exist
            similarityMatrix = createSimilarityMatrix(movies)
            print(contentBasedRecommender(similarityMatrix, movieIndex, movies))
            break
        
systemImplementation()
