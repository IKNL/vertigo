# -*- coding: utf-8 -*-
"""
auxiliaries.py
Auxiliary functions.

Created on Mon Mar 18 09:55:09 2019
@author: Arturo Moncada-Torres
arturomoncadatorres@gmail.com
"""


#%% Preliminaries
import pandas as pd
from sklearn.utils import resample


#%%
def downsample(X, y, random_state=None):
    """
    Perform class balancing downsampling.

    Parameters
    ----------
    X: Pandas data frame. Shape (n_samples, n_features)
        Feature matrix.
    y: Pandas data frame. Shape (n_samples, 1)
        Class labels. Rows are instances.
    random_state: (optional) integer
        Random seed value. Default is None (which means a np.random is used)
        Use a fixed value for reproducibility.
    
    Returns
    -------
    X_balanced, y_balanced: tuple of Pandas DataFrame. Shape (n_classes*max(n_instances_classes), n_features) and (n_classes*max(n_instances_classes), 1)
        Class-balanced input/output matrix/vector.
    """
     
    # Merge dataframe into one.
    df = pd.concat([X, y], axis=1, sort=False)
    
    # Create dataframes (a list of dataframes) for each class.
    dfs = []
    classes = list(y.T.squeeze().unique())
    
    for class_ in classes:
        df_tmp = df[df[y.columns[0]]==class_]
        dfs.append(df_tmp)
    
    counts = list(map(len, dfs))
    
    # Find the class with less instances.
    min_value = min(counts)
    min_index = counts.index(min(counts))
    min_class = classes[min_index]
    
    # Downsample all the other classes.
    iteration = 0
    for (df_, class_) in zip(dfs, classes):
        if class_ == min_class:
            # If the current class is the one with least instances,
            # do nothing.
            pass
        else:
            # Otherwise, downsample.
            df_ = resample(df_, replace=False, n_samples=min_value, random_state=random_state)
            dfs[iteration] = df_
        iteration += 1

    df_balanced = pd.concat(dfs)
    X_balanced = df_balanced.iloc[:,0:-1]
    y_balanced = pd.DataFrame(data=df_balanced.iloc[:,-1], columns=y.columns)

    return X_balanced, y_balanced