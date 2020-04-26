import sys
import os
import shutil
import time
import traceback

from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.externals import joblib


clf = None

app = Flask(__name__)

# inputs
training_data = 'data/titanic.csv'
include = ['Age', 'Sex', 'Embarked', 'Survived']
dependent_variable = 'Survived'

model_directory = 'model'
model_file_name = f'./simple_rand_forest.pkl'
model_columns_file_name = f'{model_directory}/model_columns.pkl'

first_line = "No prediciton made"
second_line = "Initital prediction not made yet"



@app.route('/')
def test():
    return render_template('index.html', first_line = first_line, second_line = second_line)
    #return render_page
    


@app.route('/predict', methods=['POST']) # Create http://host:port/predict POST end point
def predict():
    if clf:
        try:
            json_ = request.json #capture the json from POST
            query = pd.get_dummies(pd.DataFrame(json_))
            #query = query.reindex(columns=model_columns, fill_value=0)
                  
            prediction = list(clf.predict(query))
            if(len(prediction) > 1):
                global first_line
                global second_line
                first_line = "request format inconsistant"
                second_line =  "please send one pridiction at a time"
                return jsonify(first_line + " " + second_line)
                

            global first_line
            global second_line
            first_line = "A new prediction has been made"
            second_line = "your prediction is: " + prediction[0]
            #return jsonify({'prediction': [int(x) for x in prediction]})
            return jsonify(first_line + " " + second_line)

        except Exception as e:

            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        print('train first')
        return 'no model here'



if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 80

    try:
        clf = joblib.load(model_file_name)
        print('model loaded')
        

    except Exception as e:
        print("making sure")
        print(e)
        print('No model here')
        print('Train first')
        clf = None

    app.run(host='0.0.0.0', port=port, debug=False)
