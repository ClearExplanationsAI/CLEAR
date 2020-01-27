""" This is the main module for CLEAR. CLEAR takes the user-specified parameters 
    set in the module CLEAR_settings to determine the explanations to generate. 
 """

import sys
import time

import numpy as np
import CLEAR_Process_Dataset
import CLEAR_cont
import CLEAR_perturbations
import CLEAR_regression
import CLEAR_settings
import Create_sensitivity_files

start_time = time.time()
np.random.seed(1)
CLEAR_settings.init()

(X_train, X_test_sample, model, numeric_features, category_prefix, feature_list) \
    = Create_sensitivity_files.Create_sensitivity()
for neighbour_seed in range(0, CLEAR_settings.num_iterations):
    explainer = CLEAR_Process_Dataset.Create_Synthetic_Data(X_train, model, numeric_features,
                                                            category_prefix, feature_list, neighbour_seed)
    if not CLEAR_settings.LIME_comparison:
        if CLEAR_settings.multi_class:
            if CLEAR_settings.multi_class_focus == 'All':
                num_class = len(CLEAR_settings.multi_class_labels)
                multi_index = 0
            else:
                num_class = 1
                multi_index = CLEAR_settings.multi_class_labels.index(CLEAR_settings.multi_class_focus)
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
    else:
        CLEAR_cont.LIME_CLEAR(X_test_sample, explainer, feature_list,numeric_features, model)

end_time = time.time()
print("Total execution time: {}".format(end_time - start_time))
