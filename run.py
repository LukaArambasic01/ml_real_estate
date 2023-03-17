import sys
import joblib
import json
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

model = joblib.load('RealEstatePredictor.pkl')

print('Welcome\n')
print('Please select an option')
print('1) Access the predictor')
print('0) Exit')
user_input = int(input())

if user_input == 1:
    print('\n---------------------------------------------------------------------')
    print('Welcome to the Real Estate Predictor')
    file_name = input('Please provide the location of an excel file: ')
    file_data = pd.read_excel(file_name)
    with open('oe_encodes.txt', 'r') as en:
        encodes = json.load(en)
    
    temp = pd.read_csv('./RealEstate.csv')
    temp.drop(['Price', 'Address', 'Date'], axis='columns', inplace=True)
    cols = temp.columns

    X = file_data[cols]
    
    s = X['Suburb'].to_numpy()
    a = []
    for e in s:
        a.append(encodes[e])

    X['Suburb'] = a
    
    predictions = model.predict(X)
    
    print('Predictions:')
    for i in range(0, len(predictions)):
        print(f'{i}: {round(predictions[i])}$')

    en.close()
else:
    sys.exit()
