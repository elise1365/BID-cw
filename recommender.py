import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import ast
from sklearn.feature_extraction.text import TfidfVectorizer

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
    genres = movies['genres'].tolist()
    keywords = movies['keywords'].tolist()
    cast = movies['cast'].tolist()
    # Convert list of similar scores into their corresponding movie titles
    similar_movies = []
    # Filter lst by top 1000 movies ordered by revenue
    top_1000_movies = [i for i in lst if i < 1000]
    for i in top_1000_movies:
        similar_movies.append(labels[i] + '|' + genres[i] + '|' + keywords[i]+ '|' + cast[i])
        if len(similar_movies) >= 5:
            break
    # If less than 5 movies are found within top 1000, add from the overall list
    if len(similar_movies) < 5:
        remaining = 5 - len(similar_movies)
        for i in lst:
            if i >= 1000:
                similar_movies.append(labels[i] + '|' + genres[i] + '|' + keywords[i]+ '|' + cast[i])
                remaining -= 1
                if remaining == 0:
                    break
    return "If you liked '" + labels[itemIndex] + "' then you'd probably like: '" + "', '".join(similar_movies) + "'."

# Open datasets
movies = pd.read_csv(r"C:\Users\pokem\OneDrive\Documents\tmdb_5000_movies.csv")
credits = pd.read_csv(r"C:\Users\pokem\OneDrive\Documents\tmdb_5000_credits.csv")

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

#print(contentBasedRecommender(similarityMatrix, 0))
#print(contentBasedRecommender(similarityMatrix, 15))

def reduceToNames(obj):
    names = []
    if obj:
        for item in ast.literal_eval(obj):
            names.append(item['name'])
    return ' '.join(names)

# Example usage:
input_str = '[{"cast_id": 6, "character": "Woody (voice)", "credit_id": "52fe433f9251416c7500915d", "gender": 2, "id": 31, "name": "Tom Hanks", "order": 0}, {"cast_id": 9, "character": "Buzz Lightyear (voice)", "credit_id": "52fe433f9251416c75009169", "gender": 2, "id": 12898, "name": "Tim Allen", "order": 1}, {"cast_id": 17, "character": "Lotso (voice)", "credit_id": "52fe433f9251416c75009185", "gender": 2, "id": 13726, "name": "Ned Beatty", "order": 2}, {"cast_id": 8, "character": "Jessie the Yodeling Cowgirl (voice)", "credit_id": "52fe433f9251416c75009165", "gender": 1, "id": 3234, "name": "Joan Cusack", "order": 3}, {"cast_id": 7, "character": "Ken (voice)", "credit_id": "52fe433f9251416c75009161", "gender": 2, "id": 2232, "name": "Michael Keaton", "order": 4}, {"cast_id": 10, "character": "Stretch the Octopus (voice)", "credit_id": "52fe433f9251416c7500916d", "gender": 1, "id": 2395, "name": "Whoopi Goldberg", "order": 5}, {"cast_id": 11, "character": "Purple-haired doll (voice)", "credit_id": "52fe433f9251416c75009171", "gender": 1, "id": 5149, "name": "Bonnie Hunt", "order": 6}, {"cast_id": 16, "character": "Rex (voice)", "credit_id": "52fe433f9251416c75009181", "gender": 2, "id": 12900, "name": "Wallace Shawn", "order": 7}, {"cast_id": 13, "character": "Hamm (voice)", "credit_id": "52fe433f9251416c75009175", "gender": 2, "id": 7907, "name": "John Ratzenberger", "order": 8}, {"cast_id": 14, "character": "Mr. Potato Head (voice)", "credit_id": "52fe433f9251416c75009179", "gender": 2, "id": 7167, "name": "Don Rickles", "order": 9}, {"cast_id": 15, "character": "Mrs. Potato Head (voice)", "credit_id": "52fe433f9251416c7500917d", "gender": 1, "id": 61964, "name": "Estelle Harris", "order": 10}, {"cast_id": 18, "character": "Andy (voice)", "credit_id": "52fe433f9251416c75009189", "gender": 0, "id": 1116442, "name": "John Morris", "order": 11}, {"cast_id": 19, "character": "Barbie (voice)", "credit_id": "52fe433f9251416c7500918d", "gender": 1, "id": 63978, "name": "Jodi Benson", "order": 12}, {"cast_id": 20, "character": "Bonnie (voice)", "credit_id": "52fe433f9251416c75009191", "gender": 0, "id": 1096415, "name": "Emily Hahn", "order": 13}, {"cast_id": 21, "character": "Andys Mom (voice)", "credit_id": "52fe433f9251416c75009195", "gender": 1, "id": 12133, "name": "Laurie Metcalf", "order": 14}, {"cast_id": 22, "character": "Slinky Dog (voice)", "credit_id": "52fe433f9251416c75009199", "gender": 0, "id": 21485, "name": "Blake Clark", "order": 15}, {"cast_id": 23, "character": "Chatter Telephone (voice)", "credit_id": "52fe433f9251416c7500919d", "gender": 2, "id": 59357, "name": "Teddy Newton", "order": 16}, {"cast_id": 24, "character": "Trixie (voice)", "credit_id": "54bae356c3a3686c6f006840", "gender": 0, "id": 109869, "name": "Kristen Schaal", "order": 17}, {"cast_id": 25, "character": "Sarge (voice)", "credit_id": "552166b99251417be2002ef3", "gender": 2, "id": 8655, "name": "R. Lee Ermey", "order": 18}, {"cast_id": 36, "character": "Chuckles (voice)", "credit_id": "5617eddec3a3680f9a002cc7", "gender": 2, "id": 7918, "name": "Bud Luckey", "order": 19}, {"cast_id": 37, "character": "Molly (voice)", "credit_id": "5617ee409251412af8002deb", "gender": 0, "id": 97051, "name": "Beatrice Miller", "order": 20}, {"cast_id": 38, "character": "Mr. Pricklepants (voice)", "credit_id": "5617ee6d9251412af40030de", "gender": 2, "id": 10669, "name": "Timothy Dalton", "order": 21}, {"cast_id": 39, "character": "Bonnies Mom (voice)", "credit_id": "5617eeb19251412af8002df6", "gender": 1, "id": 24358, "name": "Lori Alan", "order": 22}, {"cast_id": 40, "character": "Buttercup (voice)", "credit_id": "5617eeedc3a3680fa40033a5", "gender": 2, "id": 60074, "name": "Jeff Garlin", "order": 23}, {"cast_id": 41, "character": "Twitch (voice)", "credit_id": "5617ef119251412aef002fbe", "gender": 0, "id": 167295, "name": "John Cygan", "order": 24}, {"cast_id": 42, "character": "Aliens (voice)", "credit_id": "5617ef38c3a3680fa40033c5", "gender": 2, "id": 7882, "name": "Jeff Pidgeon", "order": 25}, {"cast_id": 43, "character": "Chunk (voice)", "credit_id": "5617ef5bc3a3680fba002ffe", "gender": 2, "id": 19545, "name": "Jack Angel", "order": 26}, {"cast_id": 44, "character": "Sparks (voice)", "credit_id": "5617efaa9251415a7d0001ea", "gender": 2, "id": 157626, "name": "Jan Rabson", "order": 27}, {"cast_id": 45, "character": "Bookworm (voice)", "credit_id": "5617efccc3a3680fa40033cd", "gender": 2, "id": 21125, "name": "Richard Kind", "order": 28}, {"cast_id": 46, "character": "Sid (voice)", "credit_id": "5617eff2c3a3680f9a002d08", "gender": 2, "id": 12901, "name": "Erik von Detten", "order": 29}, {"cast_id": 47, "character": "Frog (voice)", "credit_id": "5617f0379251412af8002e0d", "gender": 2, "id": 1381777, "name": "Jack Willis", "order": 30}, {"cast_id": 48, "character": "Additional Voice (voice)", "credit_id": "5617f0519251415a7d0001fc", "gender": 2, "id": 59784, "name": "Carlos Alazraqui", "order": 31}, {"cast_id": 49, "character": "Additional Voice (voice)", "credit_id": "5617f066c3a3680fa0003282", "gender": 1, "id": 117081, "name": "Teresa Ganzel", "order": 32}, {"cast_id": 50, "character": "Additional Voice (voice)", "credit_id": "5617f087c3a3680fb6002dba", "gender": 2, "id": 84495, "name": "Jess Harnell", "order": 33}, {"cast_id": 51, "character": "Additional Voice (voice)", "credit_id": "5617f0b0c3a3680f9d002e51", "gender": 2, "id": 52699, "name": "Danny Mann", "order": 34}, {"cast_id": 52, "character": "Additional Voice (voice)", "credit_id": "5617f0d6c3a3680fa40033f1", "gender": 0, "id": 84493, "name": "Mickie McGowan", "order": 35}, {"cast_id": 53, "character": "Additional Voice (voice)", "credit_id": "5617f0f1c3a3680fb6002dc5", "gender": 1, "id": 35159, "name": "Laraine Newman", "order": 36}, {"cast_id": 54, "character": "Additional Voice (voice)", "credit_id": "5617f116c3a3680fb6002dc9", "gender": 1, "id": 1212864, "name": "Colleen OShaughnessey", "order": 37}, {"cast_id": 55, "character": "Additional Voice (voice)", "credit_id": "5617f136c3a3680fae00329e", "gender": 2, "id": 10, "name": "Bob Peterson", "order": 38}, {"cast_id": 56, "character": "Additional Voice (voice)", "credit_id": "5617f154c3a3680fba003041", "gender": 2, "id": 7960, "name": "Jerome Ranft", "order": 39}, {"cast_id": 57, "character": "Additional Voice (voice)", "credit_id": "5617f1659251412af2002fd7", "gender": 2, "id": 8, "name": "Lee Unkrich", "order": 40}, {"cast_id": 58, "character": "Additional Voice (voice)", "credit_id": "5617f17c92514141600003e3", "gender": 0, "id": 1443485, "name": "Colette Whitaker", "order": 41}, {"cast_id": 59, "character": "(voice) (uncredited)", "credit_id": "5617f289c3a3680fa400343a", "gender": 0, "id": 214701, "name": "Sherry Lynn", "order": 42}, {"cast_id": 60, "character": "(voice) (uncredited)", "credit_id": "5617f2b1c3a3680fa9002fcd", "gender": 0, "id": 86007, "name": "Jim Ward", "order": 43}, {"cast_id": 61, "character": "Bullseye / Buster / The Monkey / Pigeon (voice) (uncredited)", "credit_id": "5617f2f3c3a3680fb6002e00", "gender": 2, "id": 15831, "name": "Frank Welker", "order": 44}]'
names_only = reduceToNames(input_str)
print(names_only)

