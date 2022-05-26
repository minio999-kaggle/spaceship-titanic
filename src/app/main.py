'''
main module for app
'''
import sys
from app.preproccessing.pre_proccessing import get_df
sys.path.append('..')


PATH = "./data/train.csv"
LABEL = "Transported"

def main():
    '''
    Main function for app
    '''
    df = get_df()
    print(df.head())


if __name__ == '__main__':
    main()
