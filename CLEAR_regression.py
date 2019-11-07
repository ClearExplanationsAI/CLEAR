"""
Functions for CLEAR to create local regressions
"""

from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
import statsmodels.formula.api as sm
import tensorflow as tf
from sklearn.preprocessing import PolynomialFeatures

import CLEAR_perturbations
import CLEAR_settings

""" specify input parameters"""

kernel_type = 'Euclidean'  # sets distance measure for the neighbourhood algorithms


def Run_Regressions(X_test_sample, explainer, multi_index= None):
    """  If dataset is multiclass, then identifies w-counterfactuals and stores in boundary_df
         Labels the synthetic data and then performs the stepwise regressions. The results of the stepwise
         regression are stored in the results_df dataframe
    """

    boundary_df = []
    # label synthetic data
    if CLEAR_settings.multi_class is True:
        boundary_df = Create_boundary_df(explainer, X_test_sample, multi_index)
        explainer.master_df['prediction'] = explainer.model.predict_proba \
                                            (explainer.master_df.iloc[:,0:explainer.num_features].values)[:,multi_index]
    else:
        sensitivity_file = CLEAR_settings.case_study + '_sensitivity_' + str(CLEAR_settings.test_sample) + '.csv'
        explainer.sensit_df = pd.read_csv(CLEAR_settings.CLEAR_path + sensitivity_file)
        if CLEAR_settings.use_sklearn is True:
            y = explainer.model.predict_proba(explainer.master_df.values)
            explainer.master_df['prediction']=y[:,1]
    
        else:
            sensitivity_file = CLEAR_settings.case_study + '_sensitivity_' + str(CLEAR_settings.test_sample) + '.csv'
            explainer.sensit_df = pd.read_csv(CLEAR_settings.CLEAR_path + sensitivity_file)
            CLEAR_pred_input_func = tf.estimator.inputs.pandas_input_fn(
                x=explainer.master_df,
                batch_size=5000,
                num_epochs=1,
                shuffle=False)
            predictions = explainer.model.predict(CLEAR_pred_input_func)
    
            y = np.array([])
            for p in predictions:
                y = np.append(y, p['probabilities'][1])
            explainer.master_df['prediction'] = y

    results_df = pd.DataFrame(columns=['Reg_Score', 'intercept', 'features', 'weights',
                                       'nn_forecast', 'reg_prob', 'regression_class',
                                       'spreadsheet_data', 'local_data', 'accuracy'])
    observation_num = CLEAR_settings.first_obs
    print('Performing step-wise regressions \n')
    if multi_index is not None:
        print('Multi class ' + str(multi_index) + '\n')
    for i in range(CLEAR_settings.first_obs, CLEAR_settings.last_obs + 1):
        data_row = pd.DataFrame(columns=explainer.feature_list)
        data_row = data_row.append(X_test_sample.iloc[i], ignore_index=True)
        data_row.fillna(0, inplace=True)
        explainer = explain_data_point(explainer, data_row, observation_num, boundary_df,multi_index)
        if CLEAR_settings.with_indicator_feature is True:
            indicatorFeature_value = np.where(data_row.loc[0,CLEAR_settings.feature_with_indicator]>= CLEAR_settings.indicator_threshold, 1, 0)
            data_row['IndicatorFeature']= indicatorFeature_value

        print('Processed observation ' + str(i))
        results_df.at[i, 'features'] = explainer.features
        results_df.loc[i, 'Reg_Score'] = explainer.prediction_score
        results_df.loc[i, 'nn_forecast'] = explainer.nn_forecast
        results_df.loc[i, 'reg_prob'] = explainer.local_prob
        results_df.loc[i, 'regression_class'] = explainer.regression_class
        results_df.at[i, 'spreadsheet_data'] = explainer.spreadsheet_data
        results_df.at[i, 'local_data'] = data_row.values[0]
        results_df.loc[i, 'accuracy'] = explainer.accuracy
        results_df.loc[i, 'intercept'] = explainer.intercept
        results_df.at[i, 'weights'] = explainer.coeffs

        observation_num += 1
    if CLEAR_settings.with_indicator_feature is True:
        explainer.feature_list.append('IndicatorFeature')
    filename1 = CLEAR_settings.CLEAR_path + 'CLRreg_' + datetime.now().strftime("%Y%m%d-%H%M") + '.csv'
    results_df.to_csv(filename1)
    return (results_df, explainer,boundary_df)


def explain_data_point(explainer, data_row, observation_num, boundary_df, multi_index):
    explainer.observation_num = observation_num
    explainer.data_row = data_row
    # set to 6 for no centering when using logistic regression
    explainer.additional_weighting = 3
    # temp fix to prevent long Credit Card runs
    if CLEAR_settings.no_centering == True:
        explainer.additional_weighting = 6
    explainer.local_df = explainer.master_df.copy(deep=True)
    if CLEAR_settings.case_study in ['IRIS', 'Glass']:
        y = explainer.model.predict_proba(data_row)[:, multi_index]
    elif CLEAR_settings.use_sklearn is True:
        y = explainer.model.predict_proba(explainer.data_row.values)[0][1]  
    else:
        CLEAR_pred_input_func = tf.estimator.inputs.pandas_input_fn(
            x=data_row,
            batch_size=1,
            num_epochs=1,
            shuffle=False)
        predictions = explainer.model.predict(CLEAR_pred_input_func)
        y = np.array([])
        for p in predictions:
            y = np.append(y, p['probabilities'][1])
    
    explainer.local_df.iloc[0, 0:explainer.num_features] = data_row.iloc[0, 0:explainer.num_features]
    explainer.local_df.loc[0, 'prediction'] = y
    
    if CLEAR_settings.use_sklearn is True:
        explainer.nn_forecast = y
    else:
        explainer.nn_forecast = y[0]
    create_neighbourhood(explainer)
    if CLEAR_settings.apply_counterfactual_weights:
        add_counterfactual_rows(explainer, boundary_df, multi_index)
        if explainer.num_counterf > 0:
            adjust_neighbourhood(explainer, explainer.neighbour_df.tail(explainer.num_counterf),
                                 CLEAR_settings.counterfactual_weight)
    perform_regression(explainer)
    if CLEAR_settings.regression_type == 'logistic':
        if CLEAR_settings.case_study not in ['IRIS', 'Glass']:
            while (explainer.additional_weighting < 6 and
                   ((explainer.regression_class != explainer.nn_class) or
                    (abs(explainer.local_prob - explainer.nn_forecast) > 0.01))):
                adjust_neighbourhood(explainer, explainer.neighbour_df.iloc[0, :], 10)
                perform_regression(explainer)
        else:
            while (explainer.additional_weighting < 6 and
                   (abs(explainer.local_prob - explainer.nn_forecast) > 0.01)):
                adjust_neighbourhood(explainer, explainer.neighbour_df.iloc[0, :], 10)
                perform_regression(explainer)
    return explainer


def Create_boundary_df(explainer,X_test_sample, multi_index):
    sensitivity_file = CLEAR_settings.case_study + '_sensitivity_m' + str(multi_index) + '_' + \
                       str(CLEAR_settings.test_sample) + '.csv'

    explainer.sensit_df = pd.read_csv(CLEAR_settings.CLEAR_path + sensitivity_file)
    boundary_cols = []
    for i in range(len(X_test_sample.columns)):
        boundary_cols.append(X_test_sample.columns[i] + '_val')
        boundary_cols.append(X_test_sample.columns[i] + '_prob')
    boundary_df = pd.DataFrame(columns=boundary_cols)
    cnt = 0
    boundary_cnt = 0
    while cnt < len(explainer.sensit_df) - 1:
        current_boundary = 10000
        current_prob = 10000
        current_observ =explainer.sensit_df.loc[cnt, 'observation']
        current_feature = explainer.sensit_df.loc[cnt, 'feature']
        current_value = X_test_sample.loc[current_observ, current_feature]
        # need local data with index of current feature
        while (current_observ == explainer.sensit_df.loc[cnt, 'observation']) and \
                (current_feature == explainer.sensit_df.loc[cnt, 'feature']) and \
                (cnt < len(explainer.sensit_df) - 1):
            if (explainer.sensit_df.loc[cnt, 'newnn_class'] != explainer.sensit_df.loc[cnt + 1, 'newnn_class']) and \
                    (current_feature == explainer.sensit_df.loc[cnt + 1, 'feature']) and \
                    (explainer.sensit_df.loc[cnt, 'newnn_class'] == multi_index or
                     explainer.sensit_df.loc[cnt + 1, 'newnn_class'] == multi_index):
                new_boundary = (explainer.sensit_df.loc[cnt, 'new_value'] + explainer.sensit_df.loc[cnt + 1, 'new_value']) / 2
                if abs(new_boundary - current_value) < abs(current_boundary - current_value):
                    current_boundary = new_boundary
                    current_prob = (explainer.sensit_df.loc[cnt, 'probability'] + explainer.sensit_df.loc[cnt + 1, 'probability']) / 2
            cnt += 1
        cnt += 1
        boundary_df.loc[current_observ, current_feature + '_prob'] = current_prob
        boundary_df.loc[current_observ, current_feature + '_val'] = current_boundary
        boundary_cnt += 1
    return (boundary_df)



def adjust_neighbourhood(explainer, target_rows, num_copies):
    # add num_copies more observations
    explainer.additional_weighting += 1
    temp = pd.DataFrame(columns=explainer.neighbour_df.columns)
    temp = temp.append(target_rows, ignore_index=True)
    temp2 = explainer.neighbour_df.copy(deep=True)
    for k in range(1, num_copies):
        temp = temp.append(target_rows, ignore_index=True)
    temp3 = temp2.append(temp, ignore_index=True)
    temp3 = temp3.reset_index(drop=True)
    explainer.neighbour_df = temp3.copy(deep=True)
    if CLEAR_settings.generate_regression_files == True:
        filename1 = CLEAR_settings.CLEAR_path + 'local_' + datetime.now().strftime("%Y%m%d-%H%M%S%f") + '.csv'
        explainer.neighbour_df.to_csv(filename1)
    return explainer


def create_neighbourhood(explainer):
    # =============================================================================
    #     Generates a Neighbourhood Dataset for each observation and then performs stepwise regressions.
    #     The regressions can be polynomial and can also include interaction terms
    #     The regressions can either be either multiple or logistic regressions and
    #     can be scored using AIC, adjusted R-squared or McFadden's pseudo R-squared
    # =============================================================================


    if CLEAR_settings.neighbourhood_algorithm == 'Balanced':
        if (explainer.local_df.loc[0, 'prediction'] >= 0.1) & (explainer.local_df.loc[0, 'prediction'] <= 0.9):
            neighbour_pt1 = 0.1
            neighbour_pt2 = 0.4
            neighbour_pt3 = 0.6
            neighbour_pt4 = 0.9
        else:
            neighbour_pt1 = 0
            neighbour_pt2 = 0.4
            neighbour_pt3 = 0.6
            neighbour_pt4 = 1
        explainer.local_df.loc[
            explainer.local_df['prediction'].between(neighbour_pt1, neighbour_pt2,
                                                     inclusive=True), 'target_range'] = 1
        explainer.local_df.loc[
            explainer.local_df['prediction'].between(neighbour_pt2, neighbour_pt3,
                                                     inclusive=True), 'target_range'] = 2
        explainer.local_df.loc[
            explainer.local_df['prediction'].between(neighbour_pt3, neighbour_pt4,
                                                     inclusive=True), 'target_range'] = 3
        distances = sklearn.metrics.pairwise_distances(
            explainer.local_df.iloc[:, 0:explainer.num_features].values,
            explainer.local_df.iloc[0, 0:explainer.num_features].values.reshape(1, -1),
            metric='euclidean'
        ).ravel()
        explainer.local_df['distances'] = distances
        explainer.local_df = explainer.local_df.sort_values(by=['distances'])
        explainer.local_df = explainer.local_df.reset_index(drop=True)
        num_rows = CLEAR_settings.regression_sample_size * (neighbour_pt2 / (neighbour_pt4 - neighbour_pt1))
        temp_df = explainer.local_df[explainer.local_df['target_range'] == 1]
        temp_df = temp_df.sort_values(by=['distances'])
        temp_df = temp_df.iloc[0:int(num_rows), :]
        explainer.neighbour_df = temp_df.copy(deep=True)
        num_rows = int(CLEAR_settings.regression_sample_size * (neighbour_pt3 - neighbour_pt2) / (
                neighbour_pt4 - neighbour_pt1))
        temp_df = explainer.local_df[explainer.local_df['target_range'] == 2]
        temp_df = temp_df.sort_values(by=['distances'])
        temp_df = temp_df.iloc[0:int(num_rows), :]
        explainer.neighbour_df = explainer.neighbour_df.append(temp_df, ignore_index=True)
        num_rows = int(CLEAR_settings.regression_sample_size * (neighbour_pt4 - neighbour_pt3) / (
                neighbour_pt4 - neighbour_pt1))
        temp_df = explainer.local_df[explainer.local_df['target_range'] == 3]
        temp_df = temp_df.sort_values(by=['distances'])
        temp_df = temp_df.iloc[0:int(num_rows), :]
        explainer.neighbour_df = explainer.neighbour_df.append(temp_df, ignore_index=True)
        explainer.neighbour_df = explainer.neighbour_df.sort_values(by=['distances'])
        explainer.neighbour_df = explainer.neighbour_df.reset_index(drop=True)
        if CLEAR_settings.generate_regression_files == True:
            filename1 = CLEAR_settings.CLEAR_path + 'local_' + str(
                explainer.observation_num) + '_' + datetime.now().strftime("%Y%m%d-%H%M%S%f") + '.csv'
            explainer.neighbour_df.to_csv(filename1)
        # Creates L1 neighbourhood.selects s observations of synthetic data that are
    # nearest to the observatiom. It then checks that both classification classes are
    # sufficiently represented
    elif CLEAR_settings.neighbourhood_algorithm == 'Unbalanced':
        distances = sklearn.metrics.pairwise_distances(
            explainer.local_df.iloc[:, 0:explainer.num_features].values,
            explainer.local_df.iloc[0, 0:explainer.num_features].values.reshape(1, -1),
            metric='euclidean'
        ).ravel()
        explainer.local_df['distances'] = distances
        explainer.local_df = explainer.local_df.sort_values(by=['distances'])
        explainer.local_df = explainer.local_df.reset_index(drop=True)
        temp_df = explainer.local_df.iloc[0:int(200), :]
        explainer.neighbour_df = temp_df.copy(deep=True)
        if CLEAR_settings.generate_regression_files == True:
            filename1 = CLEAR_settings.CLEAR_path + 'local_' + str(
                explainer.observation_num) + '_' + datetime.now().strftime("%Y%m%d-%H%M%S%f") + '.csv'
            explainer.neighbour_df.to_csv(filename1)
    else:
        print('Neighbourhood Algorithm Misspecified')
    return explainer


def add_counterfactual_rows(explainer, boundary_df, multi_index):
    explainer.counterf_rows_df = pd.DataFrame(columns=explainer.neighbour_df.columns)
    if CLEAR_settings.case_study not in ['IRIS', 'Glass']:
        for feature in explainer.neighbour_df.columns[0:-3]:
            c_df = explainer.sensitivity_df[
                (explainer.sensitivity_df['observation'] == explainer.observation_num) & (
                        explainer.sensitivity_df['feature'] == feature)]
            c_df = c_df['probability'].agg(['min', 'max'])
            if (c_df['min'] <= 0.5) & (c_df['max'] > 0.5):
                old_value = explainer.neighbour_df.loc[0, feature]
                boundary = CLEAR_perturbations.Get_Counterfactual(explainer, feature, old_value,
                                                                  explainer.observation_num)
                # This is necessary for cases where the observation in sensitivity_df nearest to 50th percentile
                # is the last observation and hence is not identified by Get_Counterfactual.
                if np.isnan(boundary):
                    continue
                # estimate new_value corresponding to 50th percentile. This Gridsearch assumes only a single 50th percentile
                s1 = explainer.neighbour_df.iloc[0].copy(deep=True)
                s2 = pd.Series(s1)
                s2['target_range'] = 'counterf'
                s2['distances'] = np.nan
                s2.loc[feature] = boundary
                explainer.counterf_rows_df = explainer.counterf_rows_df.append(s2, ignore_index=True)
            explainer.num_counterf = explainer.counterf_rows_df.shape[0]
        if not explainer.counterf_rows_df.empty:
            CLEAR_pred_input_func = tf.estimator.inputs.pandas_input_fn(
                x=explainer.counterf_rows_df.iloc[:, 0:-3],
                batch_size=1,
                num_epochs=1,
                shuffle=False)
            predictions = explainer.model.predict(CLEAR_pred_input_func)
            y = np.array([])
            for p in predictions:
                y = np.append(y, p['probabilities'][1])
            explainer.counterf_rows_df['prediction'] = y
            explainer.neighbour_df = explainer.neighbour_df.append(explainer.counterf_rows_df, ignore_index=True)

    else:
        for feature in explainer.neighbour_df.columns[0:-3]:
            if boundary_df.loc[explainer.observation_num, feature + '_prob'] != 10000:
                s1 = explainer.neighbour_df.iloc[0].copy(deep=True)
                s2 = pd.Series(s1)
                s2['target_range'] = 'counterf'
                s2['distances'] = np.nan
                s2.loc[feature] = boundary_df.loc[explainer.observation_num, feature + '_val']
                explainer.counterf_rows_df = explainer.counterf_rows_df.append(s2, ignore_index=True)
        explainer.num_counterf = explainer.counterf_rows_df.shape[0]
        if not explainer.counterf_rows_df.empty:
            predictions = explainer.model.predict_proba(explainer.counterf_rows_df.iloc[:, 0:-3].values)
            explainer.counterf_rows_df['prediction'] = predictions[:, multi_index]
            explainer.neighbour_df = explainer.neighbour_df.append(explainer.counterf_rows_df, ignore_index=True)
    return explainer


def perform_regression(explainer):
    # transform neighbourhood data so that it passes through the data point to be explained
    X = explainer.neighbour_df.iloc[:, 0:explainer.num_features].copy(deep=True)
    X = X.reset_index(drop=True)
    if CLEAR_settings.regression_type in ['logistic', 'multiple']:
        decision_threshold = 0.5
        if CLEAR_settings.with_indicator_feature == True:
            indicatorFeature_value = np.where(
                (X[CLEAR_settings.feature_with_indicator] >= CLEAR_settings.indicator_threshold), 1, 0)
            X.insert(1, 'IndicatorFeature', indicatorFeature_value)
        # Take out features that are redundent or of little predictive power
        if CLEAR_settings.case_study == 'Credit Card':
            X.drop(['marDd3', 'marDd1', 'eduDd0', 'eduDd1', 'eduDd2', 'eduDd3', 'eduDd4', 'eduDd5', 'eduDd6',
                    'genDd1'], axis=1, inplace=True)
        if CLEAR_settings.no_polynomimals is True:
            poly_df = X.copy(deep=True)
        else:
            if CLEAR_settings.interactions_only is True:
                poly = PolynomialFeatures(interaction_only=True)

            else:
                poly = PolynomialFeatures(2)
            all_poss = poly.fit_transform(X)
            poly_names = poly.get_feature_names(X.columns)
            poly_names = [w.replace('^2', '_sqrd') for w in poly_names]
            poly_names = [w.replace(' ', '_') for w in poly_names]
            poly_df = pd.DataFrame(all_poss, columns=poly_names)
            if CLEAR_settings.with_indicator_feature == True:
                poly_df.drop('IndicatorFeature_sqrd', axis=1, inplace=True)
        poly_df_org_first_row = poly_df.iloc[0, :] + 0  # plus 0 is to get rid of 'negative zeros' ie python format bug
        org_poly_df = poly_df.copy(deep=True)
        # Now transform so that regression goes through the data point to be explained
        if CLEAR_settings.regression_type == 'multiple' and CLEAR_settings.no_centering == False:
            Y = explainer.neighbour_df.loc[:, 'prediction'] - explainer.nn_forecast
            poly_df = poly_df - poly_df.iloc[0, :]
        else:
            Y = explainer.neighbour_df.loc[:, 'prediction'].copy(deep=True)

        Y = Y.reset_index(drop=True)
        Y_cont = Y.copy(deep=True)
        # stepwise regression's choice of variables is restricted, but this was found to improve fidelity.
        if CLEAR_settings.case_study in ['PIMA','IRIS','Glass','BreastC']:
            if CLEAR_settings.case_study == 'PIMA':
                selected = ['1', 'BloodP', 'Skin', 'BMI', 'Pregnancy', 'Glucose', 'Insulin', 'DiabF', 'Age']
            elif CLEAR_settings.case_study == 'IRIS':
                selected = ['1', 'SepalL', 'SepalW', 'PetalL', 'PetalW']
            elif CLEAR_settings.case_study == 'Glass':
                selected = ['1', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
            elif CLEAR_settings.case_study == 'BreastC':
                selected = ['1']
            else:
                print('error in perform regression')
                exit()
            remaining = poly_df.columns.tolist()
            try:
                for x in selected:
                    remaining.remove(x)
            except:
                pass
        elif CLEAR_settings.case_study == 'Census':
            educ = ['eduDdBachelors', 'eduDdCommunityCollege', 'eduDdDoctorate', 'eduDdHighGrad', 'eduDdMasters',
                    'eduDdProfSchool']
            non_null_educ = [col for col in educ if not (X.loc[:, col] == 0).all()]
            non_null_educ = [col for col in educ if (X.loc[:, col].sum() >= 10)]
            selected = ['1', 'age', 'hoursPerWeek']
            selected = selected + non_null_educ
            # Because of the large number of features in the Census dataset, the following featurees are excluded:
            # (i) education features that are have low occurrences (ii) features which would NOT change the
            # classification of the observation, if they were independently altered.
            # These rule was adopted to improve the run time for the stepwise regressions.
            non_null_columns = [col for col in poly_df.columns[3:] if
                                ((poly_df.loc[:, col].min() == 0) & (poly_df.loc[:, col].sum() < 10))]
            poly_df.drop(non_null_columns, axis=1, inplace=True)
            remaining = poly_df.columns.tolist()
            for x in remaining:
                if x.endswith('_sqrd'):
                    if x not in ['age_sqrd', 'hoursPerWeek']:
                        poly_df.drop(x, axis=1, inplace=True)
            remaining = poly_df.columns.tolist()
        elif CLEAR_settings.case_study == 'Credit Card':
            selected = ['1', 'LIMITBAL', 'AGE', 'PAY0', 'PAY6', 'BILLAMT1', 'BILLAMT6', 'PAYAMT1', 'PAYAMT6']
            # Because of the large number of features in the Credit Card dataset, the only features that are added to
            # those selected above are features which would change the classification of the observation, if they
            # were independently altered. This rule was adopted to improve the run time for the stepwise regressions.
            for x in explainer.feature_list:
                temp_df = explainer.sensit_df[
                    (explainer.sensit_df['observation'] == explainer.observation_num) & (
                            explainer.sensit_df['feature'] == x)]
                temp_df = temp_df['probability'].agg(['min', 'max'])
                if (temp_df['min'] <= 0.5) & (temp_df['max'] > 0.5):
                    if not x in selected:
                        selected.append(x)

            remaining = poly_df.columns.tolist()
            for x in remaining:
                if x.endswith('_sqrd'):
                    if x.startswith(('mar', 'edu', 'gen', 'Indic')):
                        poly_df.drop(x, axis=1, inplace=True)
                if x.startswith('edu') and x[7:10] == 'edu':
                    poly_df.drop(x, axis=1, inplace=True)
                if x.startswith('mar') and x[7:10] == 'mar':
                    poly_df.drop(x, axis=1, inplace=True)
                if x.startswith('AGE_'):
                    poly_df.drop(x, axis=1, inplace=True)
            remaining = poly_df.columns.tolist()
            for x in selected:
                try:
                    remaining.remove(x)
                except:
                    continue

        poly_df['prediction'] = pd.to_numeric(Y_cont, errors='coerce')
        current_score, best_new_score = -1000, -1000
        while remaining and current_score == best_new_score and len(selected) < CLEAR_settings.max_predictors:
            scores_with_candidates = []
            for candidate in remaining:
                if CLEAR_settings.regression_type == 'multiple' and CLEAR_settings.no_centering == False:
                    formula = "{} ~ {}".format('prediction', ' + '.join(selected + [candidate]) + '-1')
                elif CLEAR_settings.regression_type == 'multiple' and CLEAR_settings.no_centering == True:
                    formula = "{} ~ {} ".format('prediction', ' + '.join(selected + [candidate]))
                else:
                    formula = "{} ~ {}".format('prediction', ' + '.join(selected + [candidate]))
                try:
                    if CLEAR_settings.score_type == 'aic':
                        if CLEAR_settings.regression_type == 'multiple':
                            score = sm.GLS.from_formula(formula, poly_df).fit(disp=0).aic
                        else:
                            score = sm.Logit.from_formula(formula, poly_df).fit(disp=0).aic
                        score = score * -1
                    elif CLEAR_settings.score_type == 'prsquared':
                        if CLEAR_settings.regression_type == 'multiple':
                            print("Error prsquared is not used with multiple regression")
                            exit
                        else:
                            score = sm.Logit.from_formula(formula, poly_df).fit(disp=0).prsquared
                    elif CLEAR_settings.score_type == 'adjR':
                        if CLEAR_settings.regression_type == 'multiple':
                            score = sm.GLS.from_formula(formula, poly_df).fit(disp=0).rsquared_adj
                        else:
                            print("Error Ajusted R-squared is not used with logistic regression")

                    else:
                        print('score type not correctly specified')
                        exit
                    scores_with_candidates.append((score, candidate))
                except np.linalg.LinAlgError as e:
                    if 'Singular matrix' in str(e):
                        pass
                except:
                    print("error in step regression")
            if len(scores_with_candidates) > 0:
                scores_with_candidates.sort()
                best_new_score, best_candidate = scores_with_candidates.pop()
                if current_score < best_new_score:
                    remaining.remove(best_candidate)
                    selected.append(best_candidate)
                    current_score = best_new_score
                else:
                    break
                # perfect separation
            else:
                break
        if CLEAR_settings.regression_type == 'multiple' and CLEAR_settings.no_centering == False:
            formula = "{} ~ {}".format('prediction', ' + '.join(selected) + '-1')
            selected.remove('1')
        else:
            formula = "{} ~ {}".format('prediction', ' + '.join(selected))
        try:
            if CLEAR_settings.regression_type == 'logistic':
                classifier = sm.Logit.from_formula(formula, poly_df).fit(disp=0)
            else:
                classifier = sm.GLS.from_formula(formula, poly_df).fit(disp=0)
            if CLEAR_settings.score_type == 'aic':
                explainer.prediction_score = classifier.aic
            elif CLEAR_settings.score_type == 'prsquared':
                explainer.prediction_score = classifier.prsquared
            elif CLEAR_settings.score_type == 'adjR':
                explainer.prediction_score = classifier.rsquared_adj
            else:
                print('incorrect score type')
            predictions = classifier.predict(poly_df)
            explainer.features = selected
            explainer.coeffs = classifier.params.values
            # local prob is for the target point is in class 0 . CONFIRM!
            explainer.local_prob = classifier.predict(poly_df)[0]
            if CLEAR_settings.regression_type == 'logistic':
                explainer.accuracy = (classifier.pred_table()[0][0]
                                      + classifier.pred_table()[1][1]) / classifier.pred_table().sum()
            else:
                Z = Y.copy(deep=True)
                Z[Z >= 0.5] = 2
                Z[Z < 0.5] = 1
                Z[Z == 2] = 0
                W = predictions.copy(deep=True)
                W[W >= 0.5] = 2
                W[W < 0.5] = 1
                W[W == 2] = 0
                explainer.accuracy = (W == Z).sum() / Z.shape[0]
            if CLEAR_settings.case_study in ['IRIS', 'Glass']:
                explainer.regression_class = ""  # identification of regression class requires a seperate regression for each multi class
            else:
                if explainer.local_prob >= decision_threshold:
                    explainer.regression_class = 1
                else:
                    explainer.regression_class = 0
                if explainer.nn_forecast >= decision_threshold:
                    explainer.nn_class = 1
                else:
                    explainer.nn_class = 0

            if CLEAR_settings.regression_type == 'logistic' or \
                    (CLEAR_settings.regression_type == 'multiple' and CLEAR_settings.no_centering == True):
                explainer.intercept = classifier.params[0]
                explainer.spreadsheet_data = []
                for i in range(len(selected)):
                    selected_feature = selected[i]
                    for j in range(len(classifier.params)):
                        coeff_feature = classifier.params.index[j]
                        if selected_feature == coeff_feature:
                            explainer.spreadsheet_data.append(poly_df_org_first_row.loc[selected_feature])
                explainer.untransformed_predictions = classifier.predict(org_poly_df)
            else:
                explainer.spreadsheet_data = []
                temp = 0
                explainer.intercept = +explainer.nn_forecast
                for i in range(len(selected)):
                    selected_feature = selected[i]
                    for j in range(len(classifier.params)):
                        coeff_feature = classifier.params.index[j]
                        if selected_feature == coeff_feature:
                            explainer.intercept -= poly_df_org_first_row.loc[selected_feature] * classifier.params[
                                j]
                            temp -= poly_df_org_first_row.loc[selected_feature] * classifier.params[j]
                            explainer.spreadsheet_data.append(poly_df_org_first_row.loc[selected_feature])
                            adjustment = explainer.nn_forecast - classifier.predict(poly_df_org_first_row)
                            explainer.untransformed_predictions = adjustment[0] + classifier.predict(org_poly_df)
        except:
            print(formula)
    #                input("Regression failed. Press Enter to continue...")

    else:
        print('incorrect regression type specified')
        exit
    return explainer
