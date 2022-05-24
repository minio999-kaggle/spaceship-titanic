'''
main module for app
'''

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

PATH = "./data/train.csv"
df = pd.read_csv(PATH)

features = ["Age", "Group", "NumInGroup"]
LABEL = "Transported"
mean_age = df['Age'].mean()

def splitting_id(df):
    '''
    Originally Id is in format gggg:pp where gggg is group and pp is person in group

    Parameters:
        dataframe (pandas.DataFrame): DataFrame on which to operate
    Returns:
        pandas.DataFrame
    '''
    
    df[['Group', 'NumInGroup']] = df['PassengerId'].str.split('_', 1, expand=True)
    return df

def encode_to_float(df):
    '''
    encode categorical data to float since group and num in group are objects

    Parameters:
        dataframe (pandas.DataFrame): DataFrame on which to operate
    Returns:
        pandas.DataFrame
    '''

    dfObjects = (df[features].dtypes == 'object')
    object_cols = list(dfObjects[dfObjects].index)
    ordinalEncoder = OrdinalEncoder()
    df[object_cols] = ordinalEncoder.fit_transform(df[object_cols])
    return df

def impute_age(df, value):
    '''
    Replaces Nulls in column "Age" of a dataframe with the passed value

    Parameters:
        dataframe (pandas.DataFrame): DataFrame on which to operate
        value (float): Value used for imputation
    Returns:
        pandas.DataFrame
    '''

    df['Age'] = df['Age'].fillna(value)
    return df

def transform_data(df, mean_age_value):
    '''
    Applying data cleaning functions to data sets

    Paramters:
        dataframe (pandas.DataFrame): Dataframe on which to operate
        mean_age (float): Mean age of training data set
    Retruns:
        pandas.DataFrame
    '''

    df = splitting_id(df)
    df = encode_to_float(df)
    df = impute_age(df, mean_age_value)
    return df



