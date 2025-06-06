# Imports
import bson
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from wordcloud import WordCloud
import re
from collections import Counter
from nltk.corpus import stopwords
import nltk
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from scipy.sparse import csr_matrix



###################################################################################################
#Fonctions semaine3 


def minipingpong(sparse_matrix, allusers, allgames, min_ratings_users, min_ratings_games, max_iterations=10, plot=True):
    matrix = sparse_matrix.tocoo()
    users = np.array(allusers)
    games = np.array(allgames)
    iteration = 0
    while iteration < max_iterations:
        game_ratings = np.bincount(matrix.col, minlength=len(games))
        user_ratings = np.bincount(matrix.row, minlength=len(users))

        valid_games = np.where(game_ratings >= min_ratings_games)[0]
        
        valid_users = np.where(user_ratings >= min_ratings_users)[0]
        
        keep = np.isin(matrix.row, valid_users) & np.isin(matrix.col, valid_games)
        
        new_row = matrix.row[keep]
        new_col = matrix.col[keep]
        new_data = matrix.data[keep]

        # Update row and column indices to match new dimensions
        row_mapping = {old: new for new, old in enumerate(valid_users)}
        col_mapping = {old: new for new, old in enumerate(valid_games)}
        
        new_row = np.array([row_mapping[r] for r in new_row])
        new_col = np.array([col_mapping[c] for c in new_col])
        
        # Create the new matrix
        matrix = csr_matrix((new_data, (new_row, new_col)), 
                           shape=(len(valid_users), len(valid_games)))
        # Convert back to COO for next iteration
        matrix = matrix.tocoo()
        
        # Update user and game lists
        users = users[valid_users]
        games = games[valid_games]
        iteration += 1
    
    final_matrix = matrix.tocsr()
 
    return final_matrix

###################################################################################################


def subtract_mean(cleaned_matrix):
    data_array = cleaned_matrix.toarray()
            
    user_means = []
    for i in range(data_array.shape[0]):
        user_ratings = data_array[i, :]
        non_zero_indices = user_ratings != 0
        user_ratings_non_zero = user_ratings[non_zero_indices]
        
        if len(user_ratings_non_zero) > 0:
            user_means.append(user_ratings_non_zero.mean())
        else:
            user_means.append(0)

    # Create a new matrix with normalized ratings
    normalized_data = []
    normalized_rows = []
    normalized_cols = []

    cleaned_matrix_coo = cleaned_matrix.tocoo()
    # For each non-zero entry in the original matrix, subtract user mean
    for i, j, v in zip(cleaned_matrix_coo.row, cleaned_matrix_coo.col, cleaned_matrix_coo.data):
        normalized_value = v - user_means[i]
        normalized_data.append(normalized_value)
        normalized_rows.append(i)
        normalized_cols.append(j)

    # Create new sparse matrix with normalized values
    normalized_matrix = csr_matrix(
        (normalized_data, (normalized_rows, normalized_cols)),
        shape=cleaned_matrix.shape
    )
    return normalized_matrix


###################################################################################################

def custom_train_test_split(sparse_matrix, test_size=0.2, max_user_loss=0.5, max_game_loss=0.5):

    np.random.seed(42)
    
    coo_matrix = sparse_matrix.tocoo()
    
    # Total number of ratings
    n_ratings = coo_matrix.data.size
    n_users = sparse_matrix.shape[0]
    n_games = sparse_matrix.shape[1]
    
    # Count ratings per user and per game
    user_ratings = np.bincount(coo_matrix.row, minlength=n_users)
    game_ratings = np.bincount(coo_matrix.col, minlength=n_games)
    
    # Calculate maximum allowed test ratings per user and game
    max_test_per_user = np.floor(user_ratings * max_user_loss).astype(int)
    max_test_per_game = np.floor(game_ratings * max_game_loss).astype(int)
    
    # Initialize counters for test ratings per user and game
    test_per_user = np.zeros(n_users, dtype=int)
    test_per_game = np.zeros(n_games, dtype=int)
    
    # Create a random permutation of rating indices
    rating_indices = np.random.permutation(n_ratings)
    
    # Initialize train and test masks
    train_mask = np.ones(n_ratings, dtype=bool)
    test_mask = np.zeros(n_ratings, dtype=bool)
    
    # Target number of test ratings
    n_test_target = int(test_size * n_ratings)
    n_test_selected = 0
    
    # Iterate through ratings in random order
    for idx in rating_indices:
        user = coo_matrix.row[idx]
        game = coo_matrix.col[idx]
        
        # Check if we can add this rating to test set without violating constraints
        if (test_per_user[user] < max_test_per_user[user] and 
            test_per_game[game] < max_test_per_game[game] and
            n_test_selected < n_test_target):
            
            # Add to test set
            train_mask[idx] = False
            test_mask[idx] = True
            
            # Update counters
            test_per_user[user] += 1
            test_per_game[game] += 1
            n_test_selected += 1
    
    # Create train and test matrices
    train_matrix = csr_matrix(
        (coo_matrix.data[train_mask], 
         (coo_matrix.row[train_mask], coo_matrix.col[train_mask])),
        shape=sparse_matrix.shape
    )
    
    test_matrix = csr_matrix(
        (coo_matrix.data[test_mask], 
         (coo_matrix.row[test_mask], coo_matrix.col[test_mask])),
        shape=sparse_matrix.shape
    )
    
    # Print statistics
    print(f"Split complete: {n_test_selected} ratings ({n_test_selected/n_ratings:.2%}) in test set")
    print(f"Train set: {train_matrix.nnz} ratings, Test set: {test_matrix.nnz} ratings")
    
    # Check if any users or games lost too many ratings
    users_with_ratings = np.where(user_ratings > 0)[0]
    games_with_ratings = np.where(game_ratings > 0)[0]
    
    max_user_loss_actual = np.max(test_per_user[users_with_ratings] / user_ratings[users_with_ratings])
    max_game_loss_actual = np.max(test_per_game[games_with_ratings] / game_ratings[games_with_ratings])
    
    print(f"Maximum user ratings loss: {max_user_loss_actual:.2%}")
    print(f"Maximum game ratings loss: {max_game_loss_actual:.2%}")

    return train_matrix, test_matrix, train_mask, test_mask
