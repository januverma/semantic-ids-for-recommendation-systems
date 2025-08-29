import pandas as pd
from typing import Tuple
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)

def label_from_rating(r, thr=4.0): return 1 if float(r) >= thr else 0

def temporal_split_per_user(df: pd.DataFrame, train_ratio: float = 0.8,
                           val_ratio: float = 0.1, test_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split ratings data temporally per user into train/val/test sets.

    Args:
        df: DataFrame containing ratings data with columns 'reviewerID', 'asin', 'overall', 'unixReviewTime'.
        train_ratio: Ratio for training set (default 0.8)
        val_ratio: Ratio for validation set (default 0.1)
        test_ratio: Ratio for test set (default 0.1)

    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames
    """
    # df = pd.DataFrame(all_ratings)

    # Sort by user and timestamp to ensure temporal ordering
    df = df.sort_values(['reviewerID', 'unixReviewTime'])

    # Initialize lists to store split data
    train_data = []
    val_data = []
    test_data = []

    # Group by user and split temporally
    for user_id, user_data in df.groupby('reviewerID'):
        user_ratings = user_data.sort_values('unixReviewTime')
        n_ratings = len(user_ratings)

        # Calculate split indices
        train_end = int(n_ratings * train_ratio)
        val_end = train_end + int(n_ratings * val_ratio)

        # Handle edge cases where ratios don't perfectly divide
        if train_end == 0:
            train_end = 1
        if val_end == train_end:
            val_end = min(train_end + 1, n_ratings)
        if val_end == n_ratings and n_ratings > 1:
            val_end = n_ratings - 1

        # Split the data
        train_subset = user_ratings.iloc[:train_end]
        val_subset = user_ratings.iloc[train_end:val_end]
        test_subset = user_ratings.iloc[val_end:]

        # Add to respective lists
        train_data.extend(train_subset.to_dict('records'))
        if len(val_subset) > 0:
            val_data.extend(val_subset.to_dict('records'))
        if len(test_subset) > 0:
            test_data.extend(test_subset.to_dict('records'))

    return train_data, val_data, test_data