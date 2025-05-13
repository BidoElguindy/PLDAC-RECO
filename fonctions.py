# Imports

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.sparse import csr_matrix
import pandas as pd


###################################################################################################



def minipingpong(sparse_matrix, allusers, allgames, min_ratings_users, min_ratings_games, max_iterations=10, plot=True):
  
    matrix = sparse_matrix.tocoo()
    users = np.array(allusers)
    games = np.array(allgames)
    
    print(f"Initial matrix: {matrix.shape[0]} users, {matrix.shape[1]} games, {matrix.nnz} ratings")
    
    iteration = 0
    while iteration < max_iterations:
        # Count ratings per game and per user
        game_ratings = np.bincount(matrix.col, minlength=len(games))
        user_ratings = np.bincount(matrix.row, minlength=len(users))

        # Find valid games and users
        valid_games = np.where(game_ratings >= min_ratings_games)[0]
        valid_users = np.where(user_ratings >= min_ratings_users)[0]
        
        # Keep only entries where both user and game are valid
        keep = np.isin(matrix.row, valid_users) & np.isin(matrix.col, valid_games)
        
        new_row = matrix.row[keep]
        new_col = matrix.col[keep]
        new_data = matrix.data[keep]

        # Update row and column indices to match new dimensions
        row_mapping = {old: new for new, old in enumerate(valid_users)}
        col_mapping = {old: new for new, old in enumerate(valid_games)}

        new_row = np.array([row_mapping.get(r, -1) for r in new_row])
        new_col = np.array([col_mapping.get(c, -1) for c in new_col])

        # Create the new matrix
        matrix = csr_matrix((new_data, (new_row, new_col)), 
                           shape=(len(valid_users), len(valid_games)))
        
        # Update user and game lists
        users = users[valid_users]
        games = games[valid_games]
        
        print(f"Iteration {iteration+1}: {len(users)} users, {len(games)} games, {matrix.nnz} ratings")
        
        # Check if we've converged (no more filtering needed)
        if len(valid_users) == matrix.shape[0] and len(valid_games) == matrix.shape[1]:
            print("Converged!")
            break
            
        # Convert back to COO for next iteration
        matrix = matrix.tocoo()
        iteration += 1
    
    final_matrix = matrix.tocsr()
    
    print(f"Final matrix: {final_matrix.shape[0]} users, {final_matrix.shape[1]} games, {final_matrix.nnz} ratings")
    
    return final_matrix, users, games

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
    
    print(f"Maximum user ratings loss : {max_user_loss_actual:.2%}")
    print(f"Maximum game ratings during the split: {max_game_loss_actual:.2%}")

    return train_matrix, test_matrix, train_mask, test_mask

##################################################################################################################################################################

#Baseline models


# Model 1: Global Mean Baseline
def global_mean_baseline(train_matrix,test_matrix):
    """Simplest baseline: predict global mean rating for all users and items"""
    global_mean = train_matrix.data.mean()
    test_rows, test_cols = test_matrix.nonzero()

    # For evaluating on test data
    predictions = np.full_like(test_matrix.data, global_mean)
    
    rmse = np.sqrt(mean_squared_error(predictions, test_matrix.data))
    mae = mean_absolute_error(predictions, test_matrix.data)
    r2 = r2_score(predictions, test_matrix.data)
    
    print(f"Global Mean Baseline: RMSE = {rmse:.4f}, MAE = {mae:.4f}")
    return global_mean, rmse, mae , r2 


# Model 2: User Mean Baseline
def user_mean_baseline(train_matrix, test_matrix):
    # Get user means from training data
    n_users = train_matrix.shape[0]
    test_rows, test_cols = test_matrix.nonzero()
    
    # Calculate mean rating for each user (row)
    user_means = np.zeros(n_users)
    for u in range(n_users):
        row = train_matrix.getrow(u)
        if row.nnz > 0:  # If user has any ratings
            user_means[u] = row.data.mean()
    
    # Make sure test_rows values are within bounds
    valid_indices = [i for i, row in enumerate(test_rows) if row < n_users]
    
    if len(valid_indices) < len(test_rows):
        print(f"Warning: {len(test_rows) - len(valid_indices)} test entries had out-of-bounds user indices")
    
    # Use only valid indices
    test_rows_valid = test_rows[valid_indices]
    test_cols_valid = test_cols[valid_indices] 
    test_data_valid = test_matrix.data[valid_indices]
    
    # Make predictions for test data
    predictions = np.array([user_means[row] for row in test_rows_valid])
    
    # Ensure predictions and test data have same length
    assert len(predictions) == len(test_data_valid), "Predictions and test data must have same length"
    
    rmse = np.sqrt(mean_squared_error(predictions, test_data_valid))
    mae = mean_absolute_error(predictions, test_data_valid)
    r2 = r2_score(predictions,test_data_valid )
    
    print(f"User Mean Baseline: RMSE = {rmse:.4f}, MAE = {mae:.4f}")
    return user_means, rmse, mae, r2

# Model 3: Game Mean Baseline
def item_mean_baseline(train_matrix, test_matrix):
    # Calculate mean rating for each item/game (column)
    n_items = train_matrix.shape[1]
    item_means = np.zeros(n_items)
    test_rows, test_cols = test_matrix.nonzero()
    
    # Get mean of each column (item), accounting for sparsity
    for i in range(n_items):
        col = train_matrix.getcol(i)
        if col.nnz > 0:  # If item has any ratings
            item_means[i] = col.data.mean()
    
    # Make sure test_cols values are within bounds
    valid_indices = [i for i, col in enumerate(test_cols) if col < n_items]
    
    if len(valid_indices) < len(test_cols):
        print(f"Warning: {len(test_cols) - len(valid_indices)} test entries had out-of-bounds item indices")
    
    # Use only valid indices
    test_rows_valid = test_rows[valid_indices]
    test_cols_valid = test_cols[valid_indices]
    test_data_valid = test_matrix.data[valid_indices]
    
    # Make predictions for test data
    predictions = np.array([item_means[col] for col in test_cols_valid])
    
    # Ensure predictions and test data have same length
    assert len(predictions) == len(test_data_valid), "Predictions and test data must have same length"
    
    rmse = np.sqrt(mean_squared_error(predictions, test_data_valid))
    mae = mean_absolute_error(predictions, test_data_valid)
    r2 = r2_score(predictions, test_data_valid)
  
    print(f"Item Mean Baseline: RMSE = {rmse:.4f}, MAE = {mae:.4f}")
    return item_means, rmse, mae, r2

def print_grid_search_results(results, n_top=5):
    print(f"Top {n_top} parameter combinations:")
    for i, params in enumerate(results[:n_top]):
        print(f"{i+1}. {params['params']} - RMSE: {params['mean_test_rmse']:.4f}")


##################################################################################################################################################################
# Each user has different ratings habit by subtracting each user’s average first, you remove that per-user offset. The model then concentrates on the patterns of which items a user likes more or less than their own baseline, not on whether they’re a “hard” or “easy” rater

def normalize_df(df, user_means):
    # subtract user mean (defaults to 0 if new user appears)
    df = df.copy()
    df['rating_norm'] = df.apply(
        lambda row: row['rating'] - user_means.get(row['userID'], 0),
        axis=1
    )
    return df


from surprise import Prediction 


def denormalize(predictions, user_means, min_rating=0, max_rating=10): 
    """
    Denormalizes both true (r_ui) and estimated (est) ratings in Prediction objectsd.
    """
    denormalized_predictions = []
    for pred_obj in predictions:
        user_id = pred_obj.uid
        item_id = pred_obj.iid
        true_rating_input_scale = pred_obj.r_ui
        estimated_rating_input_scale = pred_obj.est

        true_rating_denormalized = true_rating_input_scale
        estimated_rating_denormalized = estimated_rating_input_scale

        if user_id in user_means:
            true_rating_denormalized += user_means[user_id]
            true_rating_denormalized = np.clip(true_rating_denormalized, min_rating, max_rating)

            estimated_rating_denormalized += user_means[user_id]
            estimated_rating_denormalized = np.clip(estimated_rating_denormalized, min_rating, max_rating)
        
        new_pred = Prediction(
            uid=user_id,
            iid=item_id,
            r_ui=true_rating_denormalized,  # DENORMALIZED true rating
            est=estimated_rating_denormalized, # DENORMALIZED estimated rating
            details=pred_obj.details      # Preserve original details
        )
        denormalized_predictions.append(new_pred)
    return denormalized_predictions



#  Helper to extract y_true & y_pred from denorm list
def unpack(denorm_preds):
    """Extract true ratings and predictions from model results with flexible formatting."""
    y_true = []
    y_pred = []
    
    # Check first item to determine format
    sample = denorm_preds[0]
    
    # Handle different prediction formats
    if hasattr(sample, 'r_ui') and hasattr(sample, 'est'):
        # Format from Surprise KNN models (prediction objects)
        y_true = [pred.r_ui for pred in denorm_preds]
        y_pred = [pred.est for pred in denorm_preds]
    elif isinstance(sample, tuple):
        # Handle tuple format with different lengths
        if len(sample) == 5:  # (uid, iid, r_ui, est, details)
            y_true = [pred[2] for pred in denorm_preds]
            y_pred = [pred[3] for pred in denorm_preds]
        elif len(sample) == 4:  # (uid, iid, r_ui, est)
            y_true = [pred[2] for pred in denorm_preds]
            y_pred = [pred[3] for pred in denorm_preds]
        elif len(sample) == 3:  # (uid, iid, est) or other format
            y_true = [pred[1] for pred in denorm_preds]
            y_pred = [pred[2] for pred in denorm_preds]
        else:
            raise ValueError(f"Unexpected prediction tuple format: {sample}")
    else:
        raise ValueError(f"Unknown prediction format: {type(sample)}")
    
    return y_true, y_pred

##################################################################################################################################################################


def predictions_to_df(predictions):
    """Converts Surprise predictions list to DataFrame."""
    return pd.DataFrame(
        [
            {
                "userID": pred.uid,
                "itemID": pred.iid,
                "true_rating": pred.r_ui,
                "predicted_rating": pred.est,
            }
            for pred in predictions
        ]
    )