'''
main module for app
'''

import pandas as pd
from preproccessing import transform_data

PATH = "./data/train.csv"
LABEL = "Transported"

def main():
    '''
    Main function for app
    '''
    df = pd.read_csv(PATH)
    mean_age = df['Age'].mean()
    df = transform_data(df, mean_age)

if __name__ == '__main__':
    main()
