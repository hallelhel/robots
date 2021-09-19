import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.ensemble import RakelD
from sklearn.metrics import precision_score

label_names = ['trunk-flex', 'scapular-e', 'scapular-r', 'shoulder-flex', 'elbow-flex', 'distal-dys-syn']
label_indices = ['1', '2', '3', '4', '5', '9']

DATA_PATH = '../compensation_detection/Data/Soroka/cloudRequest'
OUTPUT_PATH = '../compensation_detection/Data/Soroka/code_products'


def read_movement_class_1():
    """
    read classified movement and create organized csv from it.
    This function is good for to the format of the full (old) data set labels format.
    """

    df = pd.read_excel(f'{DATA_PATH}/movement_classification.xlsx', index_col=0)
    df = df.replace(to_replace='[A-Z]+ ', value=',', regex=True)
    df = df.replace(to_replace='[A-Z]', value='', regex=True)
    df = df.dropna(how='all')
    df = df.reset_index()

    new_df = pd.DataFrame()  # if also want severity, add here
    for column in df.columns[4:-2]:
        df['experiment'] = column
        df['compensation'] = df[column].astype(str).str.replace(' ', '')

        new_df = new_df.append(df[['patient', 'experiment', 'compensation']])

    new_df['compensation'] = new_df['compensation'].astype(str).str.split(',')
    new_df['compensation'] = new_df['compensation'].apply(lambda x: list(
        set(x) & set(label_indices)) if x is not np.nan else [''])

    for name, index in zip(label_names, label_indices):
        new_df[name] = new_df['compensation'].apply(lambda x: 1 if index in x else 0)
    new_df = new_df.reset_index(drop=True)
    new_df.index = new_df['patient'] + '_' + new_df['experiment']
    new_df.to_csv(f'{OUTPUT_PATH}/patients_labels.csv')


def read_movement_class_2():
    """
    read classified movement and create organized csv from it.
    This function is good for to the format of the test(new) data set labels format.
    """

    df = pd.read_excel(F'{DATA_PATH}/movement_classification_test.xlsx', index_col=0, skiprows=1)
    df = df.replace(to_replace='[A-Z]+ ', value=',', regex=True)
    df = df.replace(to_replace='[A-Z]', value='', regex=True)
    df = df.dropna(how='all')
    df = df.reset_index().rename(columns={'index': 'patient'})

    new_df = pd.DataFrame()  # if also want severity, add here
    for column in df.columns[1:-1]:
        if '.' in column:
            df['experiment'] = str(int(column[-1]) + 1) + column[0].upper() + column[-3]
        else:
            df['experiment'] = '1' + column[0].upper() + column[-1]

        df['compensation'] = df[column]
        new_df = new_df.append(df[['patient', 'experiment', 'compensation']].dropna())

    new_df['compensation'] = new_df['compensation'].astype(str).str.split(',')
    new_df['compensation'] = new_df['compensation'].apply(lambda x: list(
        set(x) & set(label_indices)) if x is not np.nan else [''])

    for name, index in zip(label_names, label_indices):
        new_df[name] = new_df['compensation'].apply(lambda x: 1 if index in x else 0)
    new_df = new_df.reset_index(drop=True)
    new_df.index = new_df['patient'] + '_' + new_df['experiment']
    new_df.to_csv(f'{OUTPUT_PATH}/patients_labels_test.csv')



