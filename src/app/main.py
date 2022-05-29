'''
main module for app
'''
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from .preproccessing import get_df

PATH = "./data/train.csv"
features = ["Age", "Group", "NumInGroup"]
LABEL = "Transported"
N_ESTIMATORS = 300
RANDOM_STATE = 42
MAX_DEPTH = 12

def main():
    '''
    Main function for app
    '''
    df = get_df()
    X_raw = df[features]
    y = df[LABEL]
    
    k_fold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
    )

    scores = []

    for train_index, test_index in k_fold.split(X_raw):
        X_train, X_test = X_raw.loc[train_index], X_raw.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]

        clf = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                    random_state=RANDOM_STATE,
                                    max_depth=MAX_DEPTH)

        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)

        acc_score = round(accuracy_score(y_test, y_predict),3)

        print(acc_score)

        scores.append(acc_score)

    print()
    print("Average:", round(100*np.mean(scores), 1), "%")
    print("Std:", round(100*np.std(scores), 1), "%")


if __name__ == '__main__':
    main()
