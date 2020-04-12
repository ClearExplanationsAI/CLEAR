# -*- coding: utf-8 -*-
"""
This module pre-processes the selected UCI dataset and then trains a TensorFlow or sklearn model
"""
import os
import numpy as np
import pandas as pd
import sklearn
import sklearn.ensemble
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE
import CLEAR_settings


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# I have not done multiclass categorical
def Create_model_dataset():
    np.random.seed(1)
    if CLEAR_settings.sample_model == 'Credit Card':
        (X_train, X_test_sample, model, model_name, numeric_features,categorical_features, category_prefix,class_labels) = Create_Credit_Datasets()
    elif CLEAR_settings.sample_model == 'PIMA':
        (X_train, X_test_sample, model, model_name, numeric_features,categorical_features, category_prefix,class_labels) = Create_PIMA_Datasets()
    elif CLEAR_settings.sample_model == 'Adult':
        (X_train, X_test_sample, model, model_name, numeric_features,categorical_features, category_prefix,class_labels) = Create_Adult_Datasets()
    elif CLEAR_settings.sample_model == 'BreastC':
        (X_train, X_test_sample, model, model_name, numeric_features,categorical_features, category_prefix,class_labels) = Create_BreastC_Datasets()
    elif CLEAR_settings.sample_model == 'German Credit':
        (X_train, X_test_sample, model, model_name, numeric_features,categorical_features, category_prefix,class_labels) = Create_German_Datasets()
    elif CLEAR_settings.sample_model == 'IRIS':
        (X_train, X_test_sample, model, model_name, numeric_features,categorical_features, category_prefix,class_labels) = Create_IRIS_Datasets()
    return(X_train, X_test_sample, model, model_name, numeric_features,categorical_features, category_prefix,class_labels)

class Normalised_params(object):
    # Gets min and max needed for generating synthetic data
    # 'dataset_for_regressionsset' is equal to X_train for PIMA Indian Datasert
    #  X_neighborhood for Adult/Adult
    def __init__(self, X_train):
        self.m= X_train.mean()
        self.s = X_train.std()

def Create_Numeric_Multi_Datasets():
    """ Reads in IRIS Dataset and performs pre-processing
        Creates SVM model and the training and test datasets
    """
    print('Pre-processing \n')
    df = pd.read_csv(CLEAR_settings.CLEAR_path + CLEAR_settings.sample_model + '.csv')
    if CLEAR_settings.sample_model == 'IRIS':
        df.rename(columns={'SepalLength': 'SepalL', 'SepalWidth': 'SepalW',
                           'PetalLength': 'PetalL', 'PetalWidth': 'PetalW'}, inplace=True)
    num_features = df.shape[1] - 1
    X = df.iloc[:, 0:num_features]
    y = df['Outcome']
    numeric_features = X.columns.tolist()
    categorical_features =[]
    category_prefix = []
    class_labels = {0:'setosa', 1:'versicolor',2:'virginica'}
    model_name= 'IRIS'
    X_train, X_remainder, y_train, y_remainder = train_test_split(X, y, test_size=0.4, stratify=y, random_state=1)
    # standardise data using training data
    X_train = X_train.copy(deep=True)
    X_remainder = X_remainder.copy(deep=True)
    normalise = Normalised_params(X_train)
    X_train.loc[:, numeric_features] = (X_train.loc[:, numeric_features] - normalise.m) / normalise.s
    X_remainder.loc[:, numeric_features] = (X_remainder.loc[:, numeric_features] - normalise.m) / normalise.s
    X_test, X_test_sample, y_test, y_test_sample = train_test_split(X_remainder, y_remainder, test_size=.7,
                                                                    stratify=y_remainder, random_state=1)
    X_test = X_test.copy(deep=True)
    X_test_sample = X_test_sample.copy(deep=True)
    """ Create SVC in SKLearn"""
 #   model = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
    model = sklearn.svm.SVC(probability=True, gamma='scale')
    model.fit(X_train, y_train)

    print('Accuracy statistics for SVM to be explained \n')
    print(sklearn.metrics.accuracy_score(y_test, model.predict(X_test)))
    return X_train, X_test_sample, model, model_name, numeric_features,categorical_features, category_prefix, class_labels


def Create_PIMA_Datasets():
    """ Reads in PIMA Dataset and performs pre-processing
        Creates the training and test datasets
    """
    print('Pre-processing \n')
    df = pd.read_csv(CLEAR_settings.CLEAR_path + 'diabetes.csv')
    df.rename(columns={'DiabetesPedigreeFunction': 'DiabF', 'Pregnancies': 'Pregnancy',
                       'SkinThickness': 'Skin', 'BloodPressure':'BloodP'}, inplace=True)
    cols_nozero = ['Glucose', 'BloodP', 'Skin', 'Insulin', 'BMI']
    df[cols_nozero] = df[cols_nozero].replace(0, np.nan)
    df.fillna(df.mean(), inplace=True)
    num_features = df.shape[1] - 1
    X = df.iloc[:, 0:num_features]
    numeric_features = X.columns.tolist()
    categorical_features =[]
    category_prefix = []
    class_labels = {0: 'not diabetic', 1: 'diabetic'}
    model_name = 'PIMA'
    y = df['Outcome']
    (X_train, X_test_sample, model) \
        =Create_Dataset(df, X, y, numeric_features)
    return X_train, X_test_sample, model, model_name, numeric_features,categorical_features, category_prefix, class_labels



def Create_BreastC_Datasets():
    print('Pre-processing \n')
    df = pd.read_csv(CLEAR_settings.CLEAR_path + 'breast cancer.csv')
    df['diagnosis'].replace('B', 0, inplace=True)
    df['diagnosis'].replace('M', 1, inplace=True)
    for x in df.columns.tolist():
        if x.endswith('worst') or x.endswith('_se'):
            df.drop(x, axis=1, inplace=True)
    df.columns = df.columns.str.replace(r"[_]", "")
    df.columns = df.columns.str.replace("mean", "")
    df.rename(columns={'smoothness': 'smooth', 'compactness': 'compact',
                       'concave points': 'conPts', 'fractaldimension': 'fractal'}, inplace=True)
    num_features = df.shape[1] - 1
    X = df.iloc[:, 0:num_features]
    numeric_features = X.columns.tolist()
    categorical_features = []
    category_prefix = []
    class_labels = {0: 'not cancer', 1: 'cancer'}
    model_name = 'Breast Cancer'
    y = df['diagnosis']
    (X_train, X_test_sample, model) \
        =Create_Dataset(df, X, y, numeric_features)
    return X_train, X_test_sample, model, model_name, numeric_features,categorical_features, category_prefix, class_labels



def Create_Credit_Datasets():
    print('Pre-processing \n')
    df = pd.read_csv(CLEAR_settings.CLEAR_path + 'default credit cards.csv')
    categorical_features=['MARRIAGE', 'SEX', 'EDUCATION']
    category_prefix = ['mar', 'gen', 'edu']
    class_labels = {0: 'pay', 1: 'default'}
    numeric_data = df.copy()
    numeric_data.drop(['default payment', 'MARRIAGE', 'SEX', 'EDUCATION'], axis=1, inplace=True)
    numeric_features = numeric_data.columns.tolist()
    model_name = 'Credit Card'
    X = df.copy()
    X.drop(['default payment'], axis=1, inplace=True)
    X = pd.get_dummies(X, prefix=category_prefix, columns=categorical_features)
    X.columns = X.columns.str.replace(r"[_]", "Dd")
    y = df['default payment']
    (X_train, X_test_sample, model) \
        =Create_Dataset(df, X, y, numeric_features)
    return X_train, X_test_sample, model ,model_name, numeric_features,categorical_features, category_prefix, class_labels

def Create_German_Datasets():
    print('Pre-processing \n')
    df = pd.read_csv(CLEAR_settings.CLEAR_path + 'german_credit.csv')
    numeric_features = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'age',
               'existingcredits', 'peopleliable']
    categorical_features= ['existingchecking', 'credithistory', 'purpose', 'savings', 'employmentsince',
     'statussex', 'otherdebtors', 'property', 'otherinstallmentplans', 'housing', 'job',
     'telephone', 'foreignworker']
    category_prefix = ['chk', 'crh', 'pur', 'sav', 'emp','sta', 'oth', 'pro', 'oin', 'hou', 'job','tel', 'for']
    class_labels = {0: 'pay', 1:'default'}
    model_name = 'German Credit Card'
    X = df.copy()
    X.drop(['classification'], axis=1, inplace=True)
    X = pd.get_dummies(X, prefix=category_prefix,columns=categorical_features)
    X.columns = X.columns.str.replace(r"[_]", "Dd")
    y = df['classification']
    (X_train, X_test_sample, model) \
        =Create_Dataset(df, X, y, numeric_features)
    return X_train, X_test_sample, model, model_name, numeric_features,categorical_features, category_prefix, class_labels



def Create_Adult_Datasets():
    print('Pre-processing \n')
    numeric_features = ['age', 'hoursPerWeek']
    categorical_features = ['marital_status', 'occupation', 'gender', 'workclass', 'education']
    category_prefix = ['mar', 'occ', 'gen', 'wor', 'edu']
    class_labels = {0: '<=$50K', 1: '> $50K'}
    model_name = 'Adult'
    df = pd.read_csv(CLEAR_settings.CLEAR_path + 'adult.csv')
    df = df[df["workclass"] != "?"]
    df = df[df["occupation"] != "?"]
    df = df[df["native-country"] != "?"]
    df.replace(['Divorced', 'Married-AF-spouse',
                'Married-civ-spouse', 'Married-spouse-absent',
                'Never-married', 'Separated', 'Widowed'],
               ['notmarried', 'married', 'married', 'married',
                'notmarried', 'notmarried', 'notmarried'], inplace=True)
    df['education'].replace(['Preschool', '10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th'], 'dropout',
                            inplace=True)
    df['education'].replace(['HS-Grad', 'HS-grad'], 'HighGrad', inplace=True)
    df['education'].replace(['Some-college', 'Assoc-acdm', 'Assoc-voc'], 'CommunityCollege', inplace=True)
    df = df[df.race == 'White']
    df= df[df['native-country'] == 'United-States']
    # excludes 10 observations
    df = df[df['workclass'] != 'Never-worked']
    # excludes 14 observations
    df = df[df['occupation'] != 'Armed-Forces']
    # excludes 21 observations
    df = df[df['workclass'] != 'Without-pay']
    df.drop(['fnlwgt', 'educational-num', 'relationship', 'race', 'native-country', 'capital-gain', 'capital-loss'],
            axis=1, inplace=True)
    df['workclass'].replace(['Local-gov', 'State-gov', 'Federal-gov'], 'Gov', inplace=True)
    df['workclass'].replace(['Private', 'Self-emp-not-inc', 'Self-emp-inc'], 'Private', inplace=True)
    df['occupation'].replace(
        ['Craft-repair', 'Machine-op-inspct', 'Handlers-cleaners', 'Transport-moving', 'Adm-clerical',
         'Farming-fishing'], 'BlueCollar', inplace=True)
    df['occupation'].replace(['Other-service', 'Protective-serv', 'Tech-support', 'Priv-house-serv'], 'Services',
                             inplace=True)
    df['occupation'].replace(['Exec-managerial'], 'ExecManagerial', inplace=True)
    df['occupation'].replace(['Prof-specialty'], 'ProfSpecialty', inplace=True)
    df['education'].replace(['Prof-school'], 'ProfSchool', inplace=True)
    df.rename(columns={'hours-per-week': 'hoursPerWeek'}, inplace=True)
    df.rename(columns={'marital-status': 'marital_status'}, inplace=True)
    X = df.copy()
    X.drop(['income'], axis=1, inplace=True)
    X = pd.get_dummies(X, prefix=category_prefix,
                       columns= categorical_features)
    X.columns = X.columns.str.replace(r"[_]", "Dd")
    y = df["income"].apply(lambda x: ">50K" in x).astype(int)
    (X_train, X_test_sample, model) \
        =Create_Dataset(df, X, y, numeric_features)
    return X_train, X_test_sample, model, model_name, numeric_features,categorical_features, category_prefix, class_labels

def Create_Dataset(df, X, y, numeric_features):
#   use_SMOTE is True
    X_train, X_remainder, y_train, y_remainder = train_test_split(X, y, test_size=0.4, stratify=y, random_state=1)
    # standardise data using training data
    X_train = X_train.copy(deep=True)
#    if use_SMOTE is True:
#       sm = SMOTE(random_state=0)
#        X_train, y_train = sm.fit_sample(X_train, y_train)
    X_remainder = X_remainder.copy(deep=True)
    normalise = Normalised_params(X_train)
    X_train.loc[:, numeric_features] = (X_train.loc[:, numeric_features] - normalise.m) / normalise.s
    X_remainder.loc[:, numeric_features] = (X_remainder.loc[:, numeric_features] -normalise.m) / normalise.s

    X_test, X_test_sample, y_test, y_test_sample = train_test_split(X_remainder, y_remainder, test_size= .7,
                                                                  stratify=y_remainder, random_state=1)
    X_test = X_test.copy(deep=True)
    X_test_sample = X_test_sample.copy(deep=True)
    # Create the model
    # This is the size of the array we'll be feeding into our model for each example
    input_size = len(X_train.iloc[0])
    model = Sequential()
    model.add(Dense(15, input_shape=(input_size,), activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
    # Train the model
    model.fit(X_train, y_train, epochs=1, batch_size=10)
    model.summary()
    _, acc = model.evaluate(X_test,
                            y_test,
                            batch_size=100,
                            verbose=0)
    print("\nTest accuracy: %.1f%%" % (100.0 * acc))
    model.save('CLEAR_sample_model.h5')
    X_train.to_pickle('X_train')
    X_test_sample.to_pickle('X_test_sample')

    return (X_train, X_test_sample, model)


