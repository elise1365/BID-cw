# Big data code
Very basic big data code created for a university coursework.

2 Python files, one for a movie recommender system and one for a pattern classification system

[Full report](https://docs.google.com/document/d/1418zalhCxpZfbBzxRQbvckdEbhIcRhEt0VNo_b6GcYg/edit?usp=sharing)

## Movie Recommender System
This uses the TMDB 5000 dataset found [here](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata). This code recommends 5 movies to users by performing tfidf vectorizing to genres, keywords and cast and creating a similarity matrix using this information. The movies are filtered by the top 1000 highest revenue, but if recommendations are less than 5 after this, the rest of the dataset is used. 

## Pattern classification
Takes the MAFA dataset found [here](https://drive.google.com/drive/folders/1q0UwRZsNGuPtoUMFrP-U06DSYp_2E1wB) and attempts to train a model to predict if an image contains a face, even if the face has a covering e.g., mask. This uses a random forest model. Isn't the most accurate model (0.886) and is only trained on masked faces so can only correctly identify masked faces, needs to be trained on unmasked faces as well to be mor widely applicable. 
