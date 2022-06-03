'''
main module for preproccsesing data
'''

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

PATH = "./data/train.csv"

def encode_to_float(df):
    '''
    encode categorical data to float since group and num in group are objects
    Parameters:
        dataframe (pandas.DataFrame): DataFrame on which to operate
    Returns:
        pandas.DataFrame
    '''

    df_objects = (df.dtypes == 'object')
    object_cols = list(df_objects[df_objects].index)
    ordinal_encoder = OrdinalEncoder()
    df[object_cols] = ordinal_encoder.fit_transform(df[object_cols])
    return df

def scaling_features(df):
    '''
    Scaling features

    Parameters:
        df (pandas.DataFrame): Dataframe on which to operate
    Returns:
        pandas.DataFrame
    '''

    scaler = StandardScaler()
    x_train = df.drop(['Transported'], axis=1)
    scaler.fit(x_train)
    scaled_data = scaler.transform(x_train)
    scaled_data = pd.DataFrame(scaled_data, columns=x_train.columns)
    scaled_data.insert(loc=0, column='Transported', value=df['Transported'])
    return scaled_data

def impute_features(df):
    '''
    Impute missing values in features

    Parameters:
        df (pandas.DataFrame): Dataframe on which to operate
    Returns:
        pandas.DataFrame
    '''

    imputer = SimpleImputer()
    imputer.fit(df)
    imputed_df = pd.DataFrame(imputer.transform(df))
    imputed_df.columns = df.columns
    return imputed_df

def transform_data(df):
    '''
    Applying data cleaning functions to data sets

    Paramters:
        dataframe (pandas.DataFrame): Dataframe on which to operate
    Retruns:
        pandas.DataFrame
    '''
    df = encode_to_float(df)
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
