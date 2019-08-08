# -*- coding: utf-8 -*-
"""
This module pre-processes the selected UCI dataset and then trains an MLP model
"""
import numpy as np
import CLEAR_regression
import CLEAR_settings
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import lime_tabular
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
""" Specify input parameters """
#                          set to aic or adjR (adjusted R-squared)
    
    
def Create_PIMA_Datasets(): 
    """ Reads in PIMA Indians Diabetes Dataset and performs pre-processing
        Creates the training and test datasets
    """    
    print('Pre-processing \n')
    cols_nozero=['Glucose', 'BloodPressure','SkinThickness','Insulin','BMI']
    df = pd.read_csv(CLEAR_settings.CLEAR_path + 'diabetes.csv')
    df[cols_nozero]=df[cols_nozero].replace(0,np.nan)
    df.fillna(df.mean(), inplace=True)
    num_features = df.shape[1]-1
    X = df.iloc[:,0:num_features]
    feature_list=X.columns.tolist()
    y = df['Outcome']
    X_train, X_remainder, y_train, y_remainder = train_test_split(X, y, test_size=0.44, stratify=y, random_state=1)
    #standardise data using training data
    X_train=X_train.copy(deep=True)
    X_remainder=X_remainder.copy(deep=True)
    m = X_train.mean()
    s= X_train.std()
    X_train=(X_train-m)/s
    X_remainder=(X_remainder-m)/s
    
    X_test, X_remainder2, y_test, y_remainder2 = train_test_split(X_remainder, y_remainder, test_size=0.68, stratify=y_remainder, random_state=1)
    X_test=X_test.copy(deep=True)
    X_test_sample, X_test_sample2, y_test_sample, y_test_sample2 = train_test_split(X_remainder2, y_remainder2, test_size=0.5, stratify=y_remainder2, random_state=1)
    X_test_sample=X_test_sample.copy(deep=True)
    X_test_sample2=X_test_sample2.copy(deep=True)
    
    if CLEAR_settings.test_sample==2:
        X_test_sample= X_test_sample2     
    X_test_sample.reset_index(inplace=True, drop=True) 
    feature_list=X.columns.tolist()
    numeric_features = feature_list
    category_prefix=[]
    """ Create MLP model in Tensorflow"""
    
    def create_feature_cols():
      return [
        tf.feature_column.numeric_column('Pregnancies'),
        tf.feature_column.numeric_column('Glucose'),
        tf.feature_column.numeric_column('BloodPressure'),
        tf.feature_column.numeric_column('SkinThickness'),
        tf.feature_column.numeric_column('Insulin'),
        tf.feature_column.numeric_column('BMI'),
        tf.feature_column.numeric_column('DiabetesPedigreeFunction'),
        tf.feature_column.numeric_column('Age')
      ]
    
    tf.logging.set_verbosity(tf.logging.ERROR)
    input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=20,
                                                     shuffle=False,num_threads=1)
    config = tf.contrib.learn.RunConfig(tf_random_seed=234)
    model  =  tf.estimator.DNNClassifier(hidden_units=[15,15],feature_columns=create_feature_cols(),
                                         n_classes=2, config=config)
    model.train(input_fn=input_func,steps=1000)
        
    eval_input_func = tf.estimator.inputs.pandas_input_fn(
          x=X_test,
          y=y_test,
          batch_size=10,
          num_epochs=1,
          shuffle=False,
          num_threads=1)
    
    # print and store summary statistics for MLP model
    results = model.evaluate(eval_input_func)
    print('Accuracy statistics for MLP to be explained \n')
    for key in sorted(results):
            print("%s: %s" % (key, results[key]))
    return(X_train, X_test_sample,model, numeric_features,category_prefix,feature_list)

def Create_BreastC_Datasets(): 
    print('Pre-processing \n')
    df = pd.read_csv(CLEAR_settings.CLEAR_path + 'breast cancer.csv')
    df['diagnosis'].replace('B', 0, inplace = True)
    df['diagnosis'].replace('M', 1, inplace = True)
    df.rename(columns={'concave points_mean':'concave_points_mean'}, inplace = True)
    for x in df.columns.tolist():
        if x.endswith('worst') or x.endswith('_se'):
            df.drop(x, axis=1, inplace=True)
    df.columns = df.columns.str.replace(r"[_]", "")
    df.columns = df.columns.str.replace("mean", "")
    num_features = df.shape[1]-1
    X = df.iloc[:,0:num_features]
    feature_list=X.columns.tolist()
    y = df['diagnosis']
    X_train, X_remainder, y_train, y_remainder = train_test_split(X, y, test_size=0.44, stratify=y, random_state=1)
    #standardise data using training data
    X_train=X_train.copy(deep=True)
    X_remainder=X_remainder.copy(deep=True)
    m = X_train.mean()
    s= X_train.std()
    X_train=(X_train-m)/s
    X_remainder=(X_remainder-m)/s
    
    X_test, X_remainder2, y_test, y_remainder2 = train_test_split(X_remainder, y_remainder, test_size=0.81, stratify=y_remainder, random_state=1)
    X_test=X_test.copy(deep=True)
    X_test_sample, X_test_sample2, y_test_sample, y_test_sample2 = train_test_split(X_remainder2, y_remainder2, test_size=0.5, stratify=y_remainder2, random_state=1)
    X_test_sample=X_test_sample.copy(deep=True)
    X_test_sample2=X_test_sample2.copy(deep=True)
    
    if CLEAR_settings.test_sample==2:
        X_test_sample= X_test_sample2     
    X_test_sample.reset_index(inplace=True, drop=True) 
    feature_list=X.columns.tolist()
    numeric_features = feature_list
    category_prefix=[]
    """ Create MLP model in Tensorflow"""
    
    feature_columns = [
        tf.feature_column.numeric_column(name)
        for name in feature_list]
    
    tf.logging.set_verbosity(tf.logging.ERROR)
    input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=20,
                                                     shuffle=False,num_threads=1)
    config = tf.contrib.learn.RunConfig(tf_random_seed=234)
    model  =  tf.estimator.DNNClassifier(hidden_units=[15,15],feature_columns=feature_columns,
                                         n_classes=2, config=config)
    model.train(input_fn=input_func,steps=1000)
        
    eval_input_func = tf.estimator.inputs.pandas_input_fn(
          x=X_test,
          y=y_test,
          batch_size=10,
          num_epochs=1,
          shuffle=False,
          num_threads=1)
    
    # print and store summary statistics for MLP model
    results = model.evaluate(eval_input_func)
    print('Accuracy statistics for MLP to be explained \n')
    for key in sorted(results):
            print("%s: %s" % (key, results[key]))
    return(X_train, X_test_sample,model, numeric_features,category_prefix,feature_list)   


def Create_Credit_Datasets():
    print('Pre-processing \n')
    df = pd.read_csv(CLEAR_settings.CLEAR_path +'default credit cards.csv')
    
    #%%
    
    category_prefix=['mar','gen','edu']
    numeric_data =df.copy()
    numeric_data.drop(['default payment','MARRIAGE','SEX','EDUCATION'],axis =1,inplace=True)
    numeric_features=numeric_data.columns.tolist()
    
    X =df.copy()
    X.drop(['default payment'],axis =1,inplace=True)
    X=pd.get_dummies(X, prefix=category_prefix, columns=['MARRIAGE','SEX', 'EDUCATION'])
    X.columns = X.columns.str.replace(r"[_]", "Dd")
    feature_list=X.columns.tolist()
    y = df['default payment']
    X_train, X_remainder, y_train, y_remainder = train_test_split(X, y, test_size=0.4, stratify=y, random_state=1)
    #standardise data using training data
    X_train=X_train.copy(deep=True)
    X_remainder=X_remainder.copy(deep=True)
    m = X_train[numeric_features].mean()
    s= X_train[numeric_features].std()
    X_train.loc[:,numeric_features]=(X_train.loc[:,numeric_features]-m)/s
    X_remainder.loc[:,numeric_features]=(X_remainder.loc[:,numeric_features]-m)/s
    X_test, X_remainder2, y_test, y_remainder2 = train_test_split(X_remainder, y_remainder, test_size=0.9, stratify=y_remainder, random_state=1)
    X_test=X_test.copy(deep=True)
    X_neighborhood, X_remainder3, y_neighborhood, y_remainder3 = train_test_split(X_remainder2, y_remainder2, test_size=0.02, stratify=y_remainder2, random_state=1)
    X_neighborhood=X_neighborhood.copy(deep=True)
    X_test_sample, X_test_sample2, y_test_sample, y_test_sample2 = train_test_split(X_remainder3, y_remainder3, test_size=0.5, stratify=y_remainder3, random_state=1)
    X_test_sample=X_test_sample.copy(deep=True)
    X_test_sample2=X_test_sample2.copy(deep=True)
    
    if CLEAR_settings.test_sample==2:
        X_test_sample= X_test_sample2   
    my_columns = []
    
    
    
    for col in X.columns:
        my_columns.append(tf.feature_column.numeric_column(col))
    
    
    tf.logging.set_verbosity(tf.logging.ERROR)
    input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=30,num_epochs=20,
                                                     shuffle=False,num_threads=1)
    config = tf.contrib.learn.RunConfig(tf_random_seed=234)
    model  =  tf.estimator.DNNClassifier(hidden_units=[50,100],feature_columns=my_columns,
                                         n_classes=2, config=config)
    model.train(input_fn=input_func,steps=1000)
    
    eval_input_func = tf.estimator.inputs.pandas_input_fn(
          x=X_test,
          y=y_test,
          batch_size=5000,
          num_epochs=1,
          shuffle=False,
          num_threads=1)
    
    results = model.evaluate(eval_input_func)
    print('Accuracy statistics for MLP to be explained \n')
    for key in sorted(results):
            print("%s: %s" % (key, results[key]))
    ###
 
    return(X_train, X_test_sample,model, numeric_features,category_prefix,feature_list)
#This reverses the hot-encoding that was carried out for the Tensorflow model. This is necessary
# for the LIME comparison functionality (LIME reuires categorical data is unprocessed)     
def Credit_categorical(X_neighborhood):
    LIME_dataset = X_neighborhood.copy(deep=True)
    temp=LIME_dataset[['eduDd0','eduDd1','eduDd2','eduDd3','eduDd4','eduDd5','eduDd6']].idxmax(axis=1)
    temp.replace(['eduDd0','eduDd1','eduDd2','eduDd3','eduDd4','eduDd5','eduDd6'],[0,1,2,3,4,5,6],inplace= True)
    LIME_dataset['EDUCATION'] =temp
    LIME_dataset.drop(['eduDd0','eduDd1','eduDd2','eduDd3','eduDd4','eduDd5','eduDd6'],axis=1,inplace=True)
    
    temp=LIME_dataset[['marDd0','marDd1','marDd2','marDd3']].idxmax(axis=1)
    temp.replace(['marDd0','marDd1','marDd2','marDd3'],[0,1,2,3],inplace= True)
    LIME_dataset['MARRIAGE'] =temp
    LIME_dataset.drop(['marDd0','marDd1','marDd2','marDd3'],axis=1,inplace=True)
    
    temp=LIME_dataset[['genDd1','genDd2']].idxmax(axis=1)
    temp.replace(['genDd1','genDd2'],[1,2],inplace= True)
    LIME_dataset['SEX'] =temp
    LIME_dataset.drop(['genDd1','genDd2'],axis=1,inplace=True)
    feature_list=LIME_dataset.columns.tolist()
    LIME_dataset = LIME_dataset.values
    return (LIME_dataset, feature_list)



    
def Create_Census_Datasets(called_from):
    print('Pre-processing \n')
    numeric_features=['age', 'hoursPerWeek']
    category_prefix=['mar','occ','gen','work', 'edu']
    df = pd.read_csv(CLEAR_settings.CLEAR_path +'adult.csv')
    df = df[df["workclass"] != "?"]
    df = df[df["occupation"] != "?"]
    df = df[df["native-country"] != "?"]
    df.replace(['Divorced', 'Married-AF-spouse', 
                  'Married-civ-spouse', 'Married-spouse-absent', 
                  'Never-married','Separated','Widowed'],
                 ['notmarried','married','married','married',
                  'notmarried','notmarried','notmarried'], inplace = True)
    df['education'].replace(['Preschool','10th','11th','12th','1st-4th','5th-6th','7th-8th','9th'],'dropout',inplace=True)
    df['education'].replace(['HS-Grad','HS-grad'], 'HighGrad',inplace=True)
    df['education'].replace(['Some-college','Assoc-acdm', 'Assoc-voc'],'CommunityCollege',inplace=True)
    df[df.race=='White']
    df[df['native-country']=='United-States']
    #excludes 10 observations
    df= df[df['workclass']!='Never-worked']
    #excludes 14 observations
    df= df[df['occupation']!='Armed-Forces']
    #excludes 21 observations
    df=df[df['workclass']!='Without-pay']
    df.drop(['fnlwgt','educational-num','relationship','race','native-country','capital-gain','capital-loss'],axis=1, inplace=True)
    df['workclass'].replace(['Local-gov','State-gov','Federal-gov'], 'Gov',inplace=True)
    df['workclass'].replace(['Private','Self-emp-not-inc','Self-emp-inc'], 'Private',inplace=True)
    df['occupation'].replace(['Craft-repair','Machine-op-inspct','Handlers-cleaners','Transport-moving','Adm-clerical','Farming-fishing'], 'BlueCollar',inplace=True)
    df['occupation'].replace(['Other-service','Protective-serv','Tech-support','Priv-house-serv'], 'Services',inplace=True)
    df['occupation'].replace(['Exec-managerial'], 'ExecManagerial',inplace=True)
    df['occupation'].replace(['Prof-specialty'], 'ProfSpecialty',inplace=True)
    df['education'].replace(['Prof-school'], 'ProfSchool',inplace=True)
    df.rename(columns={'hours-per-week':'hoursPerWeek'}, inplace=True)
    df.rename(columns={'marital-status':'marital_status'}, inplace=True)
    X =df.copy()
    X.drop(['income'],axis =1,inplace=True)
    X=pd.get_dummies(X, prefix=category_prefix, columns=['marital_status','occupation','gender', 'workclass', 'education'])
    X.columns = X.columns.str.replace(r"[_]", "Dd")
    y=df["income"].apply(lambda x: ">50K" in x).astype(int)
    
    X_train, X_remainder, y_train, y_remainder = train_test_split(X, y, test_size=0.4, stratify=y, random_state=1)
    #standardise data using training data
    X_train=X_train.copy(deep=True)
    X_remainder=X_remainder.copy(deep=True)
    m = X_train[numeric_features].mean()
    s= X_train[numeric_features].std()
    X_train.loc[:,numeric_features]=(X_train.loc[:,numeric_features]-m)/s
    X_remainder.loc[:,numeric_features]=(X_remainder.loc[:,numeric_features]-m)/s
    X_test, X_remainder2, y_test, y_remainder2 = train_test_split(X_remainder, y_remainder, test_size=0.7, stratify=y_remainder, random_state=1)
    X_test=X_test.copy(deep=True)
    X_neighborhood, X_remainder3, y_neighborhood, y_remainder3 = train_test_split(X_remainder2, y_remainder2, test_size=0.16, stratify=y_remainder2, random_state=1)
    X_neighborhood=X_neighborhood.copy(deep=True)
    X_test_sample, X_test_sample2, y_test_sample, y_test_sample2 = train_test_split(X_remainder3, y_remainder3, test_size=0.5, stratify=y_remainder3, random_state=1)
    X_test_sample=X_test_sample.copy(deep=True)
    X_test_sample2=X_test_sample2.copy(deep=True)
    if CLEAR_settings.test_sample==2:
        X_test_sample= X_test_sample2

    feature_list=X.columns.tolist()
    my_columns = []
    for col in X.columns:
        my_columns.append(tf.feature_column.numeric_column(col))
    
    tf.logging.set_verbosity(tf.logging.ERROR)
    input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=30,num_epochs=20,
                                                     shuffle=False,num_threads=1)
    config = tf.contrib.learn.RunConfig(tf_random_seed=234)
    model  =  tf.estimator.DNNClassifier(hidden_units=[50,100],feature_columns=my_columns,
                                         n_classes=2, config=config)
    model.train(input_fn=input_func,steps=1000)
    
    eval_input_func = tf.estimator.inputs.pandas_input_fn(
          x=X_test,
          y=y_test,
          batch_size=5000,
          num_epochs=1,
          shuffle=False,
          num_threads=1)
    results = model.evaluate(eval_input_func)
    print('Accuracy statistics for MLP to be explained \n')
    for key in sorted(results):
            print("%s: %s" % (key, results[key]))
            
    X_test.reset_index(inplace=True, drop=True)      
    return(X_train, X_test_sample,model,numeric_features,category_prefix,feature_list)

def Adult_categorical(X_neighborhood):
    LIME_dataset = X_neighborhood.copy(deep=True)
    temp=LIME_dataset[['eduDdBachelors','eduDdCommunityCollege','eduDdDoctorate','eduDdHighGrad','eduDdMasters','eduDdProfSchool','eduDddropout']].idxmax(axis=1)
    temp.replace(['eduDdBachelors','eduDdCommunityCollege','eduDdDoctorate','eduDdHighGrad','eduDdMasters','eduDdProfSchool','eduDddropout'],[0,1,2,3,4,5,6],inplace= True)
    LIME_dataset['EDUCATION'] =temp
    LIME_dataset.drop(['eduDdBachelors','eduDdCommunityCollege','eduDdDoctorate','eduDdHighGrad','eduDdMasters','eduDdProfSchool','eduDddropout'],axis=1,inplace=True)
    
    temp=LIME_dataset[['occDdBlueCollar','occDdExecManagerial','occDdProfSpecialty','occDdSales','occDdServices']].idxmax(axis=1)
    temp.replace(['occDdBlueCollar','occDdExecManagerial','occDdProfSpecialty','occDdSales','occDdServices'],[0,1,2,3,4],inplace= True)
    LIME_dataset['Occupation'] =temp
    LIME_dataset.drop(['occDdBlueCollar','occDdExecManagerial','occDdProfSpecialty','occDdSales','occDdServices'],axis=1,inplace=True)
    
    temp=LIME_dataset[['workDdGov','workDdPrivate']].idxmax(axis=1)
    temp.replace(['workDdGov','workDdPrivate'],[1,2],inplace= True)
    LIME_dataset['Work'] =temp
    LIME_dataset.drop(['workDdGov','workDdPrivate'],axis=1,inplace=True)
    
    temp=LIME_dataset[['marDdmarried','marDdnotmarried']].idxmax(axis=1)
    temp.replace(['marDdmarried','marDdnotmarried'],[1,2],inplace= True)
    LIME_dataset['MARRIAGE'] =temp
    LIME_dataset.drop(['marDdmarried','marDdnotmarried'],axis=1,inplace=True)
    
    temp=LIME_dataset[['genDdFemale','genDdMale']].idxmax(axis=1)
    temp.replace(['genDdFemale','genDdMale'],[1,2],inplace= True)
    LIME_dataset['SEX'] =temp
    LIME_dataset.drop(['genDdFemale','genDdMale'],axis=1,inplace=True)
    feature_list=LIME_dataset.columns.tolist()
    LIME_dataset = LIME_dataset.values
    return (LIME_dataset, feature_list)


def Create_Neighbourhoods(X_train,X_test_sample,model,\
                          numeric_features,category_prefix,feature_list,neighbour_seed):                  
    np.random.seed(neighbour_seed)   
    if (CLEAR_settings.case_study== 'PIMA Indians Diabetes') and (CLEAR_settings.test_sample==1):
        sensitivity_df = pd.read_csv(CLEAR_settings.CLEAR_path +'PIMA_sensitivity.csv')     
    elif (CLEAR_settings.case_study== 'PIMA Indians Diabetes') and (CLEAR_settings.test_sample==2):
        sensitivity_df = pd.read_csv(CLEAR_settings.CLEAR_path +'Final_diabetes_sensitivity2.csv')
    elif (CLEAR_settings.case_study== 'Credit Card') and (CLEAR_settings.test_sample==1):
        sensitivity_df = pd.read_csv(CLEAR_settings.CLEAR_path +'Credit_card_sensitivity.csv')   
    elif (CLEAR_settings.case_study== 'Credit Card') and (CLEAR_settings.test_sample==2):
        sensitivity_df = pd.read_csv(CLEAR_settings.CLEAR_path +'Credit_card_sensitivity2.csv')   
    elif (CLEAR_settings.case_study== 'Census') and (CLEAR_settings.test_sample==1):
        sensitivity_df = pd.read_csv(CLEAR_settings.CLEAR_path +'Final_cens_sensitivity1.csv')   
    elif (CLEAR_settings.case_study== 'Census') and (CLEAR_settings.test_sample==2):
        sensitivity_df = pd.read_csv(CLEAR_settings.CLEAR_path +'Final_cens_sensitivity2.csv')  
    elif (CLEAR_settings.case_study== 'BreastC') and (CLEAR_settings.test_sample==1):
        sensitivity_df = pd.read_csv(CLEAR_settings.CLEAR_path +'BreastC_sensitivity1.csv')   
    elif (CLEAR_settings.case_study== 'BreastC') and (CLEAR_settings.test_sample==2):
        sensitivity_df = pd.read_csv(CLEAR_settings.CLEAR_path +'BreastC_sensitivity2.csv')      
    else:
        print('Test sample incorrectly specified')
        exit()      
    if CLEAR_settings.LIME_comparison == True:
        if CLEAR_settings.case_study in ['BreastC','PIMA Indians Diabetes']:
            explainer = lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_list,\
                                            discretize_continuous=False,kernel_width=CLEAR_settings.LIME_kernel,\
                                            sample_around_instance=True)
        elif  CLEAR_settings.case_study == 'Credit Card':
                  LIME_dataset, feature_list = Credit_categorical(X_train)    
                  explainer = lime_tabular.LimeTabularExplainer(LIME_dataset, feature_names=feature_list,\
                                            feature_selection='forward_selection',discretize_continuous=False, \
                                            categorical_features =[20,21,22],kernel_width=CLEAR_settings.LIME_kernel,\
                                            sample_around_instance=True)
        elif  CLEAR_settings.case_study == 'Census':
                    LIME_dataset, feature_list = Adult_categorical(X_train)    
                    explainer = lime_tabular.LimeTabularExplainer(LIME_dataset, feature_names=feature_list,\
                                            feature_selection='forward_selection',discretize_continuous=False,\
                                            categorical_features =[2,3,4,5,6],kernel_width=CLEAR_settings.LIME_kernel,\
                                            sample_around_instance=True)
        else:
            print('Error in LIME comparison: Create Neighbourhood method')  
    else:    
        explainer = CLEAR_regression.CLEARExplainer(X_train, model,numeric_features,\
                                            category_prefix,feature_list,sensitivity_df)     
    return(X_test_sample,explainer,sensitivity_df,feature_list,numeric_features,model)

