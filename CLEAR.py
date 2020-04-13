""" This is the main module for CLEAR. CLEAR can either be run with:
(a) one of the sample models/datasets provided in CLEAR_sample_models_datasets.py .To do this run
    Run_CLEAR_with_sample_model()
(b) with a user created model and datasets. In this case run
     Run_CLEAR(X_train, X_test_sample, model, model_name, numeric_features, categorical_features, category_prefix, class_labels)
     An example of the required inputs is provided at the bottom of this module.
CLEAR's input parameters are specified in CLEAR_settings.py
 """

import time
import pandas as pd
import tensorflow as tf
import numpy as np
import CLEAR_sample_models_datasets
import CLEAR_perturbations
import CLEAR_regression
import CLEAR_settings
import CLEAR_sensitivity_files


def Run_CLEAR_with_sample_model():
    CLEAR_settings.init()
    (X_train, X_test_sample, model, model_name, numeric_features,categorical_features, category_prefix, class_labels)\
        =CLEAR_sample_models_datasets.Create_model_dataset()
    CLEAR_Main(X_train, X_test_sample, model, model_name, numeric_features, categorical_features, category_prefix,class_labels)
    return()

def Run_CLEAR(X_train, X_test_sample, model, model_name, numeric_features, categorical_features, categorical_prefix, class_labels):
    CLEAR_settings.init()
    feature_list= X_train.columns.to_list()
    if len(categorical_features) != len(categorical_prefix):
        print(" Number of features in 'categorical features' and 'category prefix' inputs are not the same")
        exit()
    for i in numeric_features:
        if i not in feature_list:
            print(" Some of the inputs in 'numeric features' are not in X_train")
    #for i in categorical_prefix:
    CLEAR_Main(X_train, X_test_sample, model, model_name, numeric_features, categorical_features, categorical_prefix,class_labels)
    return()

def CLEAR_Main(X_train, X_test_sample, model, model_name, numeric_features, categorical_features, category_prefix,class_labels):
    start_time = time.time()
    np.random.seed(1)
    CLEAR_sensitivity_files.Create_sensitivity(X_train, X_test_sample, model, numeric_features, category_prefix,class_labels)
    for neighbour_seed in range(0, CLEAR_settings.num_iterations):
        explainer = CLEAR_regression.Create_Synthetic_Data(X_train, model, model_name, numeric_features, categorical_features,
                                                            category_prefix, class_labels, neighbour_seed)
        if len(class_labels) > 2:
            if CLEAR_settings.multi_class_focus == 'All':
                num_class = len(class_labels)
                multi_index = 0
            else:
                num_class = 1
                multi_index =[k for k, v in class_labels.items() if v == CLEAR_settings.multi_class_focus][0]
            for c in range(num_class):
                (results_df, explainer,single_regress,boundary_df) = CLEAR_regression.Run_Regressions(X_test_sample, explainer, multi_index)
                (nn_df, miss_df) = CLEAR_perturbations.Calculate_Perturbations(explainer, results_df, boundary_df, multi_index)
                if (multi_index == 0 and CLEAR_settings.multi_class_focus == 'All') or num_class == 1 :
                    nncomp_df = nn_df.copy(deep=True)
                    missing_log_df= miss_df.copy(deep=True)
                else:
                    nncomp_df=nncomp_df.append(nn_df,sort=False)
                    missing_log_df=missing_log_df.append(miss_df,sort=False)
                multi_index +=1
        else:
            (results_df, explainer,single_regress,boundary_df) = CLEAR_regression.Run_Regressions(X_test_sample, explainer)
            (nncomp_df, missing_log_df) = CLEAR_perturbations.Calculate_Perturbations(explainer, results_df, boundary_df)
        CLEAR_perturbations.Summary_stats(nncomp_df, missing_log_df)
        if CLEAR_settings.first_obs == CLEAR_settings.last_obs:
            CLEAR_perturbations.Single_prediction_report(results_df, nncomp_df, single_regress, explainer)
    end_time = time.time()
    print("Total execution time: {}".format(end_time - start_time))
    return()

if __name__ == "__main__":
    # X_train = pd.read_pickle('C:/Users/adamp/Dropbox/Warwick/X_train_Adult')
    # X_test_sample = pd.read_pickle('C:/Users/adamp/Dropbox/Warwick/X_test_sample_Adult')
    # model = tf.keras.models.load_model('C:/Users/adamp/Dropbox/Warwick/CLEAR_Adult.h5')
    # model_name = 'Adult'
    # numeric_features = ['age', 'hoursPerWeek']
    # categorical_features = ['marital_status', 'occupation', 'gender', 'workclass', 'education']
    # category_prefix = ['mar', 'occ', 'gen', 'wor', 'edu']
    # class_labels = {0: '<=$50K', 1: '> $50K'}
    # Run_CLEAR(X_train, X_test_sample, model, model_name, numeric_features, categorical_features, category_prefix, class_labels)

    Run_CLEAR_with_sample_model()