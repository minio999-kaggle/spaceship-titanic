'''
main module for app
'''

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

PATH = "./data/train.csv"
df = pd.read_csv(PATH)

features = ["Age", "Group", "NumInGroup"]
LABEL = "Transported"

# Since passengerID in original is combination of Group and number in group
# in  format like gggg_pp we are going to split them
df[['Group', 'NumInGroup']] = df['PassengerId'].str.split('_', 1, expand=True)

# We are going to encode Group and NumInGroup to numbers
dfObjects = (df[features].dtypes == 'object')
object_cols = list(dfObjects[dfObjects].index)
ordinalEncoder = OrdinalEncoder()
df[object_cols] = ordinalEncoder.fit_transform(df[object_cols])

# to fill in missing values we need mean age
mean_age = df['Age'].mean()
df['Age'] = df['Age'].fillna(mean_age)
