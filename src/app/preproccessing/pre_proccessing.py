'''
main module for preproccsesing data
'''

from sklearn.impute import SimpleImputer
import pandas as pd

PATH = "./data/train.csv"
features = ["Age", "Group", "NumInGroup"]

def impute_features(df):
    '''
    Impute missing values in features

    Parameters:
        df (pandas.DataFrame): Dataframe on which to operate
    Returns:
        pandas.DataFrame
    '''
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(df)
    imputed_df = pd.DataFrame(imputer.transform(df))
    imputed_df.columns = df.columns
    return imputed_df

def scaling_features(df):
    '''
    Scaling features

    Parameters:
        df (pandas.DataFrame): Dataframe on which to operate
    Returns:
        pandas.DataFrame
    '''
    scaler = StandardScaler()
    scaler.fit(df)
    scaled_df = pd.DataFrame(scaler.transform(df))
    scaled_df.columns = df.columns
    return scaled_df

def transform_data(df):
    '''
    Applying data cleaning functions to data sets

    Paramters:
        dataframe (pandas.DataFrame): Dataframe on which to operate
    Retruns:
        pandas.DataFrame
    '''

    df = impute_features(df)
    df = scaling_features(df)
    return df

def get_df():
    '''
    Sharing dataframe after aplying preprocessing

    Returns:
        pandas.DataFrame
    '''
    df = pd.read_csv(PATH)
    df = transform_data(df)
    return df
