import csv
import io

from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS
import json
from Stroke_code import feature_Selection
import pandas as pd

app = Flask(__name__)
CORS(app)
print("running")
# currently we use only 3 sensors
DATA_PATH = '../compensation_detection/Data/Soroka'

LABEL = 'PatientID'
TASK = 'taskID'
features = ['wrist_X', 'wrist_Y', 'wrist_Z', 'shoulder_X', 'shoulder_Z', 'elbow_X']

input_fold = 'organized_train_data'
output_fold = 'code_products'
DATA_KIND = 'train'

# comment this line after data creation
# feature_Selection.create_tsfresh_data(features, input_fold, output_fold, LABEL, TASK, data_kind=DATA_KIND)

# PatientID,taskID,idx
X_train, Y_train = feature_Selection.read_tsfresh_data(
    x_path=f'{DATA_PATH}/{output_fold}/tsfresh_features_{DATA_KIND}.csv',
    y_path=f'{DATA_PATH}/{output_fold}/patients_labels.csv')

model = feature_Selection.run_rakel_full(X_train, Y_train)




@app.route('/testdata')
def use_model():
    data = request.json
    from Stroke_code.read_stroke import read_data
    from Stroke_code.labels_creation import read_movement_class_1, read_movement_class_2
    in_test_path = '../compensation_detection/Data/Soroka/cloudRequest'
    out_test_path = '../compensation_detection/Data/Soroka/organized_test_data_cloud'
    # with test

    # read_movement_class_1()
    # read_movement_class_2()

    f = request.files['data_file']
    if not f:
        return "No file"

    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    df = pd.read_csv(stream)
    df = df * 1000

    ######## addition to algorithm ##############

    df.to_csv('C:/Users/ronit/Downloads/robots/compensation_detection/Data/Soroka/cloudRequest/P02/P02_high_3_full.csv')
    stream.close()
    try:
        read_data(in_test_path, out_test_path)
    except:
        return "Data comprehension problem, retry to create data file", 400
    DATA_KIND = 'test'
    input_fold = 'organized_test_data_cloud'
    output_fold = 'prod1'
    print("lets go")
    try:
        feature_Selection.create_tsfresh_data(features, input_fold, output_fold, LABEL, TASK, data_kind=DATA_KIND)
    except:
        return "Data comprehension problem, retry to create data file", 400
    print("features created")
    X_test = feature_Selection.read_tsfresh_request(
        x_path=f'{DATA_PATH}/{output_fold}/tsfresh_features_{DATA_KIND}.csv')
    ans = feature_Selection.predict_singal_compensation(model, X_test)
    print("done")
    print(ans)
    print(type(ans))
    return {"results": ans.tolist()}, 200


if __name__ == '__main__':
    app.run(debug=True)
