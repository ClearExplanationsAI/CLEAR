import re
from datetime import datetime

import numpy as np
import pandas as pd

import CLEAR_Process_Dataset,CLEAR_regression,CLEAR_perturbations
import CLEAR_settings


# This runs CLEAR using the neighbourhood dataset generation and regression algorithms of LIME
def LIME_CLEAR(X_test_sample, explainer,feature_list,numeric_features, model):
    explainer.feature_list = feature_list
    explainer.numeric_features = numeric_features
    explainer.model = model
    results_df = pd.DataFrame(columns=['multi_index', 'Reg_score', 'intercept', 'features', 'weights',
                                       'nn_forecast', 'reg_prob', 'regression_class',
                                       'spreadsheet_data', 'local_data', 'accuracy'])
    observation_num = CLEAR_settings.first_obs
    feature_list = X_test_sample.columns.tolist()
    multi_index = 0
    for i in range(CLEAR_settings.first_obs, CLEAR_settings.last_obs + 1):
        data_row = pd.DataFrame(columns=feature_list)
        data_row = data_row.append(X_test_sample.iloc[i], ignore_index=True)
        data_row.fillna(0, inplace=True)

        if CLEAR_settings.case_study == 'Credit Card':
            LIME_datarow, LIME_features = CLEAR_Process_Dataset.Credit_categorical(data_row)
            LIME_datarow = LIME_datarow.flatten()
            lime_out = explainer.explain_instance(LIME_datarow, explainer.model,
                                                  num_features=CLEAR_settings.max_predictors,
                                                  num_samples=CLEAR_settings.LIME_sample)
        elif CLEAR_settings.case_study == 'Census':
            LIME_datarow, LIME_features = CLEAR_Process_Dataset.Adult_categorical(data_row)
            LIME_datarow = LIME_datarow.flatten()
            lime_out = explainer.explain_instance(LIME_datarow, explainer.model,
                                                  num_features=CLEAR_settings.max_predictors,
                                                  num_samples=CLEAR_settings.LIME_sample)


        elif CLEAR_settings.multi_class is True:
            lime_out = explainer.explain_instance(X_test_sample.iloc[i].values, model,
                                                  top_labels=len(CLEAR_settings.multi_class_labels),
                                                  num_features=CLEAR_settings.max_predictors,
                                                  num_samples=CLEAR_settings.LIME_sample)
        else:
            lime_out = explainer.explain_instance(X_test_sample.iloc[i].values, model,
                                                  num_features=CLEAR_settings.max_predictors,
                                                  num_samples=CLEAR_settings.LIME_sample)

        coeffs = np.asarray([x[1] for x in lime_out.local_exp[1]])
        feature_idx = np.asarray([x[0] for x in lime_out.local_exp[1]])
        features = [lime_out.domain_mapper.exp_feature_names[x] for x in feature_idx]
        if CLEAR_settings.case_study == 'Credit Card':
            str1 = ','.join(features)
            str1 = str1.replace("MARRIAGE=", "marDd")
            str1 = str1.replace("EDUCATION=", "eduDd")
            str1 = str1.replace("SEX=", "genDd")
            features = str1.split(",")
        if CLEAR_settings.case_study == 'Census':
            str1 = ','.join(features)
            rep = {'EDUCATION=0': 'eduDdBachelors', 'EDUCATION=1': 'eduDdCommunityCollege',
                   'EDUCATION=2': 'eduDdDoctorate', 'EDUCATION=3': 'eduDdHighGrad',
                   'EDUCATION=4': 'eduDdMasters', 'EDUCATION=5': 'eduDdProfSchool',
                   'EDUCATION=6': 'eduDddropout', 'Occupation=0': 'occDdBlueCollar',
                   'Occupation=1': 'occDdExecManagerial', 'Occupation=2': 'occDdProfSpecialty',
                   'Occupation=3': 'occDdSales', 'Occupation=4': 'occDdServices',
                   'Work=1': 'workDdGov', 'Work=2': 'workDdPrivate',
                   'MARRIAGE=1': 'marDdmarried', 'MARRIAGE=2': 'marDdnotmarried',
                   'SEX=1': 'genDdFemale', 'SEX=2': 'genDdMale'}
            rep = dict((re.escape(k), v) for k, v in rep.items())
            pattern = re.compile("|".join(rep.keys()))
            str1 = pattern.sub(lambda m: rep[re.escape(m.group(0))], str1)
            features = str1.split(",")
        if CLEAR_settings.multi_class is True:
            if i > CLEAR_settings.first_obs:
                results_df.drop(results_df.index[0])
            if CLEAR_settings.multi_class_focus == 'All':
                num_class = len(CLEAR_settings.multi_class_labels)
                multi_index = 0
            else:
                num_class = 1
                multi_index = CLEAR_settings.multi_class_labels.index(CLEAR_settings.multi_class_focus)
            for c in range(num_class):
                coeffs = np.asarray([x[1] for x in lime_out.local_exp[c]])
                feature_idx = np.asarray([x[0] for x in lime_out.local_exp[c]])
                features = [lime_out.domain_mapper.exp_feature_names[x] for x in feature_idx]
                Write_results(i, multi_index, features, data_row, lime_out, coeffs, results_df)
                boundary_df = CLEAR_regression.Create_boundary_df(explainer, X_test_sample, multi_index)
                (nn_df, miss_df) = CLEAR_perturbations.Calculate_Perturbations(explainer, results_df, boundary_df, multi_index)
                if (i==CLEAR_settings.first_obs and multi_index == 0):
                    nncomp_df = nn_df.copy(deep=True)
                    missing_log_df= miss_df.copy(deep=True)
                    cum_results_df= results_df.copy(deep= True)
                else:
                    nncomp_df=nncomp_df.append(nn_df,sort=False)
                    missing_log_df=missing_log_df.append(miss_df,sort=False)
                    cum_results_df = cum_results_df.append(results_df.tail(1), sort=False)
                multi_index+=1
        else:

            Write_results(i, 1, features, data_row, lime_out, coeffs, results_df)
            sensitivity_file = CLEAR_settings.case_study + '_sensitivity_' + str(CLEAR_settings.test_sample) + '.csv'
            explainer.sensit_df = pd.read_csv(CLEAR_settings.CLEAR_path + sensitivity_file)
            (nn_df, miss_df) = CLEAR_perturbations.Calculate_Perturbations(explainer, results_df, [])
            if i == CLEAR_settings.first_obs:
                nncomp_df = nn_df.copy(deep=True)
                missing_log_df = miss_df.copy(deep=True)
                cum_results_df = results_df.copy(deep=True)
            else:
                nncomp_df = nncomp_df.append(nn_df, sort=False)
                missing_log_df = missing_log_df.append(miss_df, sort=False)
                cum_results_df = cum_results_df.append(results_df.tail(1), sort=False)




        observation_num += 1
    CLEAR_perturbations.Summary_stats(nncomp_df, missing_log_df)
    filename1 = CLEAR_settings.CLEAR_path + 'LIME_' + datetime.now().strftime("%Y%m%d-%H%M") + '.csv'
    results_df.to_csv(filename1)
    filename2 = CLEAR_settings.CLEAR_path + 'Cum_LIME_' + datetime.now().strftime("%Y%m%d-%H%M") + '.csv'
    cum_results_df.to_csv(filename2)
    """ 
    Counterfactual perturbations are now calculated and stored
        in the nncomp_df dataframe. If CLEAR calculates a perturbation
        that is infeasible, then the details of the perturbation
        are stored in the missing_log_df dataframe. CLEAR will classify
        a perturbation as being infeasible if it is outside 'the feasibility
        range' it calculates for each variable.
    """


#    (nncomp_df, missing_log_df) = CLEAR_perturbations.Calculate_Perturbations(explainer, results_df, boundary_df)
    return


def Write_results(i, multi_index, features, data_row, lime_out, coeffs, results_df):
    print('Processed observation ' + str(i) + ' class ' + str(multi_index))
    results_df.loc[i, 'multi_index'] = multi_index
    results_df.at[i, 'features'] = features
    results_df.loc[i, 'Reg_score'] = lime_out.score[multi_index]
    results_df.loc[i, 'nn_forecast'] = lime_out.predict_proba[multi_index]
    results_df.loc[i, 'reg_prob'] = lime_out.local_pred[multi_index][0]
    results_df.loc[i, 'regression_class'] = 'x'
    results_df.at[i, 'spreadsheet_data'] = 'x'
    results_df.at[i, 'local_data'] = data_row.values[0]
    results_df.loc[i, 'accuracy'] = 'x'
    results_df.loc[i, 'intercept'] = lime_out.intercept[multi_index]
    results_df.at[i, 'weights'] = coeffs
    return results_df
