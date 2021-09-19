from skmultilearn.ensemble import RakelD
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn import model_selection
from sklearn.linear_model import RidgeClassifierCV
# from sktime.datasets import load_arrow_head  # univariate dataset
# from sktime.transformations.panel.rocket import Rocket

from Stroke_code.pre_processing import *
from Stroke_code.utils import *
from Stroke_code.create_tsfresh_features import *
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.metrics import roc_curve, auc
import seaborn as sn
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold

DATA_PATH = '../compensation_detection/Data/Soroka'

LABELS = ['trunk-flex', 'scapular-e', 'scapular-r', 'shoulder-flex', 'elbow-flex', 'distal-dys-syn']

# full feature list
features = ['wrist_X', 'wrist_Y', 'wrist_Z', 'index_X', 'index_Y',
            'index_Z', 'thumb_X', 'thumb_Y', 'thumb_Z', 'data_cup_X', 'data_cup_Y',
            'data_cup_Z', 'shoulder_X', 'shoulder_Y', 'shoulder_Z', 'elbow_X',
            'elbow_Y', 'elbow_Z', 's1_X', 's1_Y', 's1_Z', 's2_X', 's2_Y', 's2_Z',
            'hum_X', 'hum_Y', 'hum_Z', 'for_X', 'for_Y', 'for_Z', 'rad_X', 'rad_Y', 'rad_Z']

LABEL = 'PatientID'
TASK = 'taskID'

# Seed value
seed_value = 42
input_fold = 'organized_train_data'
output_fold = 'code_products'
DATA_KIND = 'train'
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)


def run_rakel(X, y):
    """
    Run algorithm on given data set in Leave-One-Out evaluation.
    :param X: data features
    :param y: labels
    """
    # number of samples and number of features
    n_samples, n_features = X.shape

    # since we want to evaluate only on the patients (and not the controls), the control group will be in the train
    # set. probably in the future it is better to do it differently since it assume that patients are before controls
    # in the array.
    n_samples = 522

    # split data into 29 folds (number of patients)
    fold = model_selection.KFold(n_splits=int(n_samples / 18), shuffle=False)

    # perform evaluation on classification task
    y_predict_all = np.zeros((n_samples, y.shape[1]))

    for idx, (train_idx, test_idx) in enumerate(fold.split(X[:n_samples])):
        print(idx)
        # if labelset_size=1 it is like running random forest on each compensation
        clf = RakelD(base_classifier=RandomForestClassifier(n_estimators=1000), labelset_size=2)

        clf.fit(X[list(train_idx) + list(range(522, y.shape[0]))], y[list(train_idx) + list(range(522, y.shape[0]))])

        # predict the class labels of test data
        y_predict = clf.predict_proba(X[test_idx]).toarray()
        y_predict_all[test_idx, :] = y_predict

    lr_auc = roc_auc_score(y[:n_samples], y_predict_all)
    print('ROC AUC=%.3f' % lr_auc)

    print('ROC with micro AVG', roc_auc_score(y[:n_samples], y_predict_all, average='micro'))

    print_pred(np.round(y_predict_all), y[:n_samples])
    print_roc(y_predict_all, y[:n_samples])
    precision_recall_graph(y[:n_samples], y_predict_all, LABELS, method='baseline_Patients_LOO_hybrid')







#Testing of a model based of DNN
def model_dnn(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(units=40, input_dim=n_inputs, activation='relu'))
    model.add(Dense(55, input_dim=40, activation='softplus'))
    model.add(Dense(25, input_dim=55, activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def run_DNN(X, y):
    print(X.shape)
    print(y.shape)
    print(X.shape[1])
    n_samples = 522
    results = list()
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    # define evaluation procedure

    cv = RepeatedKFold(n_splits=int(n_samples / 18), n_repeats=2, random_state=1)
    # enumerate folds
    y_predict_all = np.zeros((X.shape[0], y.shape[1]))
    for train_ix, test_ix in cv.split(X[:n_samples]):
        # prepare data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        a, b = load_arrow_head(split="test", return_X_y=True)
        c, d = load_arrow_head(split="train", return_X_y=True)

        rocket = Rocket(num_kernels=10000, random_state=111)
        rocket.fit(X)
        X_train_transform = rocket.transform(X)
        print(X_train_transform.shape)

        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
        classifier.fit(X_train_transform, y_train)
        X_test_transform = rocket.transform(X_test)
        results.append(classifier.score(X_test_transform, y_test))
        print(results)

    print('Accuracy: %.3f (%.3f)' % (np.mean(results), np.std(results)))

    # history = model.fit(trainX, trainy, epochs=300, verbose=2, validation_data=(testX, testy), shuffle = False)

    # # recall: tp / (tp + fn)
    # recall = recall_score(testy, yhat_classes)
    # print('Recall: %f' % recall)

#ROC GRAPH
def print_roc(Y_pred, Y_true):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for idx in range(Y_pred.shape[1]):
        y_true_i = Y_true[:, idx]
        y_pred_i = Y_pred[:, idx]
        fpr[idx], tpr[idx], _ = roc_curve(y_true_i, y_pred_i)
        roc_auc[idx] = auc(fpr[idx], tpr[idx])
        pyplot.plot(fpr[idx], tpr[idx], marker='.', label=LABELS[idx])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_true.ravel(), Y_pred.ravel())
    pyplot.plot(fpr["micro"], tpr["micro"], marker='.', label='micro avg ROC')
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print('AUC micro average : ', roc_auc)

    pyplot.plot([0, 1], [0, 1], 'k--')
    pyplot.legend()
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title('RAkEL ROC')
    pyplot.show()

#Corrlaition between features ans sensors
def print_corr_matrix(X):
    data = {'0': X[:, 0], '1': X[:, 1], '2': X[:, 2], '3': X[:, 3], '4': X[:, 4], '5': X[:, 5], '6': X[:, 6],
            '7': X[:, 7], '8': X[:, 8], '9': X[:, 9], '10': X[:, 10], '11': X[:, 11], '12': X[:, 12], '13': X[:, 13],
            '14': X[:, 14], '15': X[:, 15], '16': X[:, 16], '17': X[:, 17], '18': X[:, 18], '19': X[:, 19],
            '20': X[:, 20], '21': X[:, 21], '22': X[:, 22], '23': X[:, 23], '24': X[:, 24], '25': X[:, 25],
            '26': X[:, 26], '27': X[:, 27], '28': X[:, 28], '29': X[:, 29]}

    cols = np.arange(start=0, stop=30, step=1)
    list_cols = list(cols)
    print(list_cols)

    new_list = []
    for i in list_cols:
        new_list.append(str(i))
    print(new_list)
    df = pd.DataFrame(data, columns=new_list)

    corrMatrix = df.corr()

    corrMatrix = corrMatrix.round(2)
    # print(corrMatrix.shape[0])
    # for r in range(corrMatrix.shape[0]):
    #     for c in range(corrMatrix.shape[1]):
    #         corrMatrix[r][c] = "{:.2f}".format(round(corrMatrix[r][c], 2))

    sn.heatmap(corrMatrix, annot=True)
    plt.show()

#Shap features extractions
def select_features(X,y):
    n_samples = 522



    y_pred_all = np.zeros((n_samples, y.shape[1]))


    rocket = Rocket()  # by default, ROCKET uses 10,000 kernels

    rocket.fit(X)
    X_train_transform = rocket.transform(X)
    y2 = y[:, 1]
    y3 = y[:, 2]
    y4 = y[:, 3]
    y5 = y[:, 4]
    y6 = y[:, 5]

    classifier = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
    classifier.fit(X_train_transform, y2)

    shap_values = shap.TreeExplainer(classifier).shap_values(X_train_transform)
    shap.summary_plot(shap_values, X_train_transform, plot_type="bar",title="scapular_e")

    classifier = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
    classifier.fit(X_train_transform, y3)

    shap_values = shap.TreeExplainer(classifier).shap_values(X_train_transform)
    shap.summary_plot(shap_values, X_train_transform, plot_type="bar",title="scapular_r")

    classifier = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
    classifier.fit(X_train_transform, y4)

    shap_values = shap.TreeExplainer(classifier).shap_values(X_train_transform)
    shap.summary_plot(shap_values, X_train_transform, plot_type="bar",title="shoulder_flex")

    classifier = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
    classifier.fit(X_train_transform, y5)

    shap_values = shap.TreeExplainer(classifier).shap_values(X_train_transform)
    shap.summary_plot(shap_values, X_train_transform, plot_type="bar",title="elbow_flex")

    classifier = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
    classifier.fit(X_train_transform, y6)

    shap_values = shap.TreeExplainer(classifier).shap_values(X_train_transform)
    shap.summary_plot(shap_values, X_train_transform, plot_type="bar",title="distal_dys_syn")


def union_(lista, listb):
    list_res = []
    for i in range(len(lista)):
        if lista[i] == 1 or listb[i] == 1:
            list_res.append(1)
        else:
            list_res.append(0)
    return list_res

#with union
def print_acc(Y_pred, Y_true):
    acc = 0
    for row in range(Y_pred.shape[0]):
        tp = np.sum(np.round(np.clip(Y_true[row] * Y_pred[row], 0, 1)))
        union = np.sum(union_(Y_true[row], Y_pred[row]))
        acc += tp / union
    return acc / Y_pred.shape[0]


def print_pred(Y_pred, Y_true):
    """
    Print prediction by precision and recall measures
    :param Y_pred:predicted (0 or 1 )
    :param Y_true: true labels
    """

    for idx in range(Y_pred.shape[1]):
        y_true_i = Y_true[:, idx]
        y_pred_i = Y_pred[:, idx]
        tp = np.sum(np.round(np.clip(y_true_i * y_pred_i, 0, 1)))
        fp = np.sum(np.round(np.clip(y_pred_i - y_true_i, 0, 1)))
        # fn = np.sum(np.round(np.clip(y_true_i - y_pred_i, 0, 1)), axis=1)
        # calculate precision
        p = tp / (tp + fp)
        print(f'precision of {LABELS[idx]} : {p}')

    print(f'precision_score mean micro:{precision_score(Y_true, Y_pred, average="micro")}')
    print(f'precision_score mean macro:{precision_score(Y_true, Y_pred, average="macro")}')
    print(f'recall_score mean micro:{recall_score(Y_true, Y_pred, average="micro")}')
    print(f'recall_score mean macro:{recall_score(Y_true, Y_pred, average="macro")}')


def read_tsfresh_data(x_path, y_path, merge='left'):
    """
    Read data after tsfresh feature creation and return features and labels data frames/
    :param x_path: path to tsfresh features
    :param y_path: path to labels
    :param merge: left or inner join. if there are rows without labels that need to be dropped - use inner join.
    if there are rows without labels that need to be classified as zeroes (like control group) use left join.
    :return: df with feature, data frame with labels (same indices).
    """

    data = pd.read_csv(x_path, index_col=0)
    data_labels = pd.read_csv(f'{y_path}', index_col=0).iloc[:, -6:]

    data = data.merge(data_labels, how=merge, left_on='taskID', right_index=True)
    data.drop(['PatientID'], axis=1, inplace=True)
    data = data.fillna(0)
    datax = data.drop(LABELS, axis=1)
    datax = datax.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

    datay = data.loc[:, ['taskID'] + LABELS]
    print(datax.shape)
    return datax.set_index('taskID'), datay.set_index('taskID')


def read_tsfresh_request(x_path):
    data = pd.read_csv(x_path, index_col=0)

    # data = data.merge(data_labels, how=merge, left_on='taskID', right_index=True)
    data.drop(['PatientID'], axis=1, inplace=True)
    data = data.fillna(0)
    # datax = data.drop(LABELS, axis=1)
    datax = data.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

    print(datax.shape)
    return datax.set_index('taskID')


def run_rakel_full(X_train, y_train):
    """
    Run rakel on all train set without LOO, and predict on the test set.
    :param X_train: np array with training data
    :param y_train: np array with training labels
    :param X_test: np array with test features
    :param y_test: np array with test labels
    """

    clf = RakelD(
        base_classifier=RandomForestClassifier(n_estimators=2500),
        labelset_size=2
    )

    clf.fit(X_train, y_train)
    return clf



'''
    predict model- get precetage
    optional: change tresh hold
'''
def predict_singal_compensation(model, test):
    # predict the class labels of test data
    y_predict = model.predict_proba(test).toarray()
    print("********************")
    print(y_predict)

    print("NEW************")
    print(y_predict)


    return np.round(y_predict)


def create_tsfresh_data(full_column_list, input_path, output_path, label='PatientID', task='taskID', data_kind='train'):
    """
    full process of reading the data, preprocessing and feature creation
    :param full_column_list: list of columns to extract
    :param input_path: path to input directory
    :param output_path: path to output directory
    :param label: patient id
    :param task: task id which is the unique experiment id (with patient id)
    :param data_kind: train \ test
    """
    import time

    start = time.time()
    preprocessed_df = preprocess(full_column_list + ['Time'], label, task=task, handle_nan=True, with_controls=True,
                                 time_series_folder=f'{DATA_PATH}/{input_path}')

    preprocessed_df.to_csv(f'{DATA_PATH}/{output_path}/movements_after_preprocess_{data_kind}.csv')

    data = pd.read_csv(f'{DATA_PATH}/{output_path}/movements_after_preprocess_{data_kind}.csv', index_col=0)
    tsfresh_features_df = tsfresh_features(data, 'Time', 'PatientID', 'taskID', 'idx')
    tsfresh_features_df.to_csv(f'{DATA_PATH}/{output_path}/tsfresh_features_{data_kind}.csv')

    end = time.time()
    print(end - start)


