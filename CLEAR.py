""" This is the main module for CLEAR. CLEAR takes the user-specified parameters 
    set in the module CLEAR_settings to determine the explanations to generate. 
 """
 
import sys
import numpy as np
import CLEAR_cont, Create_sensitivity_files, CLEAR_regression
import CLEAR_settings, CLEAR_Process_Dataset, CLEAR_perturbations
import time

start_time = time.time()
np.random.seed(1)
CLEAR_settings.init()

(X_train, X_test_sample, model, numeric_features, category_prefix, feature_list) \
 = Create_sensitivity_files.Create_sensitivity()
for neighbour_seed in range(0, CLEAR_settings.Num_iterations):
    (X_test_sample, explainer, sensitivity_df, feature_list, numeric_features, model)\
     = CLEAR_Process_Dataset.Create_Neighbourhoods(
             X_train, X_test_sample, model, numeric_features,
             category_prefix, feature_list, neighbour_seed)

    if CLEAR_settings.LIME_comparison is False:
        (results_df, regression_obj)=CLEAR_regression.Run_Regressions(X_test_sample, \
        explainer, feature_list)
        nncomp_df=CLEAR_perturbations.Calculate_Perturbations(explainer, results_df, sensitivity_df,\
                                                             feature_list, numeric_features, model)
        if CLEAR_settings.first_obs == CLEAR_settings.last_obs:
            CLEAR_perturbations.Single_prediction_report(results_df, nncomp_df, regression_obj,feature_list)
    elif CLEAR_settings.LIME_comparison is True:
        CLEAR_cont.LIME_CLEAR(X_test_sample, explainer, sensitivity_df, \
                              feature_list, numeric_features, model)
    else:
        print('Evaluation type misspecified')
        sys.exit()

end_time = time.time()
print("Total execution time: {}".format(end_time - start_time))
