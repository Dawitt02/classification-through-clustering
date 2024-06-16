import pandas as pd
from sklearn.datasets import make_blobs
import numpy as np

'------------------------------------ Information about generate test data --------------------------------------------'
'''
This module will generate a test data set if you want to use random data.
Please set the parameters "n_samples", "n_features", "centers", "random_state" as you want.
'''
'----------------------------------------------------------------------------------------------------------------------'


def create_test_dataset(data_format):
    if data_format == 'bool':
        # Generate Boolean test data
        data = np.random.choice([False, True], size=(100, 10))
        data_frame = pd.DataFrame(data)
    elif data_format == 'num':
        # Generate Numeric test data
        n_samples = 301
        n_features = 2
        centers = 301
        random_state = 4300
        data, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=random_state)
        # Create a DataFrame with an additional ID column at the beginning
        data_frame = pd.DataFrame(data, columns=['Feature1', 'Feature2'])
        # Add an ID column
        data_frame.insert(0, 'ID', range(1, len(data_frame) + 1))
    else:
        raise ValueError("Invalid format. Please choose 'bool' or 'num'.")

    return data_frame
