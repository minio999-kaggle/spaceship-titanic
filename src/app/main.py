'''
main module for app
'''
from .preproccessing import get_df


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
