import numpy as np
import pandas as pd
from app.preproccessing import encode_to_float, scaling_features, impute_features, transform_data, get_df

def test_encode_to_float():
    """
    Test encode_to_float function
    """
    test_data = ['test', 'test2', 'test3']
    df = pd.DataFrame(test_data)
    df = encode_to_float(df)
    assert df.dtypes == np.float64

def test_scaling_features():
    """
    Test scaling_features function
    """
    test_data = [1,2,3,4,5,6]
    df = pd.DataFrame(test_data)
    df = scaling_features(df)
    assert df.dtypes == np.float64

def test_impute_features():
    """
    Test impute_features function
    """
    test_data = [np.nan, 2.01, 3.02, np.nan, np.nan, np.nan]
    df = pd.DataFrame(test_data)
    df = impute_features(df)
    assert df.dtypes == np.float64

def test_transorm_data():
    """
    Test transform_data function
    """
    test_data = [1,2,3,4,5,6]
    df = pd.DataFrame(test_data)
    df = transform_data(df)
    assert df.dtypes == np.float64

def test_get_df():
    """
    Test get_df function
    """
    df = get_df()
    assert df.dtypes == np.float64