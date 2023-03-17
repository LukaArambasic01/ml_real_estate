import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBRegressor
import joblib
import json


data = pd.read_csv('./RealEstate.csv')

data.dropna(axis='rows', subset=['Price'], inplace=True)

y = data['Price']
X = data.drop(['Price', 'Address', 'Date'], axis='columns')

before = X['Suburb'].unique()
oe = OrdinalEncoder()
X[['Suburb']] = oe.fit_transform(X[['Suburb']])
after = X['Suburb'].unique()

# OrdinalEncoder dictionary
oe_dict = {}
for i in range(0, len(before)):
    oe_dict[before[i]] = after[i]

with open('oe_encodes.txt', 'w') as fp:
    json.dump(oe_dict, fp)


numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
categorical_cols = [col for col in X.columns if (X[col].dtype == 'object')]

numerical_transformer = SimpleImputer(strategy = 'constant')
categorial_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorial_transformer, categorical_cols)
])

model = XGBRegressor(n_estimators=1000, learning_rate=0.06, random_state=0)

main_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

main_model.fit(X, y)

joblib.dump(main_model, 'RealEstatePredictor.pkl')