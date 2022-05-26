'''
main module for app
'''
import sys
import pandas as pd
from app.preproccessing.pre_proccessing import transform_data
sys.path.append('..')


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
