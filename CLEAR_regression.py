"""
Functions for CLEAR to create local regressions

Outstanding work - perform_regression() accuracy & decision boundaries for multi-class
"""

from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
import statsmodels.api as sm
import tensorflow as tf
import copy
from sklearn.preprocessing import PolynomialFeatures
from scipy.signal import argrelextrema
import CLEAR_settings
from scipy.spatial.distance import cdist

""" specify input parameters"""

kernel_type = 'Euclidean'  # sets distance measure for the neighbourhood algorithms


class CLEARSingleRegression(object):
    # Contains features specific to a particular b-counterfactual
    def __init__(self,observation_num, data_row):
        self.features = 0
        self.prediction_score = 0
        self.nn_forecast = 0
        self.local_prob = 0
        self.regression_class = 0
        self.spreadsheet_data = 0
        self.untransformed_predictions = 0
        self.data_row = 0
        self.accuracy = 0
        self.intercept = 0
        self.coeffs = 0
        self.observation_num = observation_num
        self.data_row = data_row
        self.additional_weighting = 0
        self.local_df = 0
        self.nn_forecast = 0
        self.neighbour_df = 0


def Run_Regressions(X_test_sample, explainer, multi_index=None):
    """  If dataset is multiclass, then identifies b-counterfactuals and stores in multiClassBoundary_df
         Labels the synthetic data and then performs the stepwise regressions. The results of the stepwise
         regression are stored in the results_df dataframe
    """
    # label synthetic data
    if CLEAR_settings.multi_class is True:
        explainer.master_df['prediction'] = explainer.model.predict_proba \
                                                (explainer.master_df.iloc[:, 0:explainer.num_features].values)[:,
                                            multi_index]
    else:
        sensitivity_file = CLEAR_settings.case_study + '_sensitivity_' + str(CLEAR_settings.test_sample) + '.csv'
        explainer.sensit_df = pd.read_csv(CLEAR_settings.CLEAR_path + sensitivity_file)
        sensitivity_file = CLEAR_settings.case_study + '_catSensitivity_' + str(CLEAR_settings.test_sample) + '.csv'
        if len(explainer.category_prefix) != 0:
            explainer.catSensit_df = pd.read_csv(CLEAR_settings.CLEAR_path + sensitivity_file)
        if CLEAR_settings.use_sklearn is True:
            y = explainer.model.predict_proba(explainer.master_df.values)
            explainer.master_df['prediction'] = y[:, 1]

        else:
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

    multiClassBoundary_df = []
    if CLEAR_settings.multi_class is True:
        multiClassBoundary_df = get_multclass_boundaries(explainer, X_test_sample, multi_index)
    get_counterfactuals(explainer, multiClassBoundary_df, X_test_sample, multi_index)
    results_df = pd.DataFrame(columns=['Reg_Score', 'intercept', 'features', 'weights',
                                       'nn_forecast', 'reg_prob', 'regression_class',
                                       'spreadsheet_data', 'local_data', 'accuracy'])
    print('Performing step-wise regressions \n')
    if multi_index is not None:
        print('Multi class ' + str(multi_index) + '\n')
    for i in range(CLEAR_settings.first_obs, CLEAR_settings.last_obs + 1):
        data_row = pd.DataFrame(columns=explainer.feature_list)
        data_row = data_row.append(X_test_sample.iloc[i], ignore_index=True)
        data_row.fillna(0, inplace=True)
        explainer, single_regress = explain_data_point(explainer, data_row, i, multiClassBoundary_df, multi_index)
        print('Processed observation ' + str(i))
        results_df.at[i, 'features'] = single_regress.features
        results_df.loc[i, 'Reg_Score'] = single_regress.prediction_score
        results_df.loc[i, 'nn_forecast'] = single_regress.nn_forecast
        results_df.loc[i, 'reg_prob'] = single_regress.local_prob
        results_df.loc[i, 'regression_class'] = single_regress.regression_class
        results_df.at[i, 'spreadsheet_data'] = single_regress.spreadsheet_data
        results_df.at[i, 'local_data'] = data_row.values[0]
        results_df.loc[i, 'accuracy'] = single_regress.accuracy
        results_df.loc[i, 'intercept'] = single_regress.intercept
        results_df.at[i, 'weights'] = single_regress.coeffs

    filename1 = CLEAR_settings.CLEAR_path + 'CLRreg_' + datetime.now().strftime("%Y%m%d-%H%M") + '.csv'
    results_df.to_csv(filename1)
    return (results_df, explainer, single_regress, multiClassBoundary_df)


def explain_data_point(explainer, data_row, observation_num, boundary_df, multi_index):
    single_regress = CLEARSingleRegression(observation_num,data_row)
    # This is for centering when using logistic regression
    if CLEAR_settings.regression_type == 'logistic':
        if CLEAR_settings.centering is True:
            single_regress.additional_weighting = 0
        else:
            single_regress.additional_weighting = 2
    single_regress.local_df = explainer.master_df.copy(deep=True)
    y = forecast_data_row(explainer, data_row, multi_index)
    single_regress.local_df.iloc[0, 0:explainer.num_features] = data_row.iloc[0, 0:explainer.num_features]
    single_regress.local_df.loc[0, 'prediction'] = y
    if CLEAR_settings.use_sklearn is True:
        if type(y) == np.ndarray:
            y = y[0]
        single_regress.nn_forecast = y
    else:
        single_regress.nn_forecast = y[0]
    create_neighbourhood(explainer, single_regress)
    temp_df=explainer.counterf_rows_df[explainer.counterf_rows_df.observation == single_regress.observation_num].copy(deep=True)
    if (CLEAR_settings.apply_counterfactual_weights) and (temp_df.empty is False):
        temp_df=temp_df.drop(['observation', 'feature'], axis=1)
        single_regress.neighbour_df = single_regress.neighbour_df.append(temp_df, ignore_index=True, sort=False)
        if explainer.num_counterf > 0:
            adjust_neighbourhood(single_regress, single_regress.neighbour_df.tail(explainer.num_counterf),
                                 CLEAR_settings.counterfactual_weight)
    # if no centering == False and regression type - logistic then initially add 19 rows of the observation
    # that is to be explained to the neighbourhood dataset. This in effect is the same as adding a weighting of 20
    if (CLEAR_settings.regression_type == 'logistic') and (CLEAR_settings.centering is True):
        adjust_neighbourhood(single_regress, single_regress.neighbour_df.iloc[0, :], 19)
    perform_regression(explainer, single_regress)
    if CLEAR_settings.regression_type == 'logistic':
        if CLEAR_settings.case_study not in ['IRIS', 'Glass']:
            while (single_regress.additional_weighting < 2 and
                   ((single_regress.regression_class != single_regress.nn_class) or
                    (abs(single_regress.local_prob - single_regress.nn_forecast) > 0.01))):
                single_regress.additional_weighting += 1
                adjust_neighbourhood(single_regress, single_regress.neighbour_df.iloc[0, :], 10)
                perform_regression(explainer, single_regress)
        else:
            while (single_regress.additional_weighting < 2 and
                   (abs(single_regress.local_prob - single_regress.nn_forecast) > 0.01)):
                explainer.additional_weighting += 1
                adjust_neighbourhood(single_regress, single_regress.neighbour_df.iloc[0, :], 10)
                perform_regression(explainer, single_regress)
    return explainer, single_regress


def forecast_data_row(explainer, data_row, multi_index):
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
    return (y)


def get_multclass_boundaries(explainer, X_test_sample, multi_index):
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
        current_observ = explainer.sensit_df.loc[cnt, 'observation']
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
                new_boundary = (explainer.sensit_df.loc[cnt, 'new_value'] + explainer.sensit_df.loc[
                    cnt + 1, 'new_value']) / 2
                if abs(new_boundary - current_value) < abs(current_boundary - current_value):
                    current_boundary = new_boundary
                    current_prob = (explainer.sensit_df.loc[cnt, 'probability'] + explainer.sensit_df.loc[
                        cnt + 1, 'probability']) / 2
            cnt += 1
        cnt += 1
        boundary_df.loc[current_observ, current_feature + '_prob'] = current_prob
        boundary_df.loc[current_observ, current_feature + '_val'] = current_boundary
        boundary_cnt += 1
    return (boundary_df)


def adjust_neighbourhood(single_regress, target_rows, num_copies):
    # add num_copies more observations
    temp = pd.DataFrame(columns=single_regress.neighbour_df.columns)
    temp = temp.append(target_rows, ignore_index=True)
    temp2 = single_regress.neighbour_df.copy(deep=True)
    for k in range(1, num_copies):
        temp = temp.append(target_rows, ignore_index=True)
    temp3 = temp2.append(temp, ignore_index=True)
    temp3 = temp3.reset_index(drop=True)
    single_regress.neighbour_df = temp3.copy(deep=True)
    if CLEAR_settings.generate_regression_files == True:
        filename1 = CLEAR_settings.CLEAR_path + 'local_' + datetime.now().strftime("%Y%m%d-%H%M%S%f") + '.csv'
        single_regress.neighbour_df.to_csv(filename1)
    return single_regress


def create_neighbourhood(explainer, single_regress):
    # =============================================================================
    #  Generates a Neighbourhood Dataset for each observation that will be used
    #  in the stepwise regressions. The algorithms select synthetic obsevations
    #  that are 'near' to the observation to be explained. Distance is calculated
    #  using a weighted sum of the Euclidean distance for numerical features and
    #  the Jaccard distance for categorical features.
    #  ============================================================================

    if CLEAR_settings.neighbourhood_algorithm == 'Balanced':
        if (single_regress.local_df.loc[0, 'prediction'] >= 0.1) & (
                single_regress.local_df.loc[0, 'prediction'] <= 0.9):
            neighbour_pt1 = 0.1
            neighbour_pt2 = 0.4
            neighbour_pt3 = 0.6
            neighbour_pt4 = 0.9
        else:
            neighbour_pt1 = 0
            neighbour_pt2 = 0.4
            neighbour_pt3 = 0.6
            neighbour_pt4 = 1
        single_regress.local_df.loc[
            single_regress.local_df['prediction'].between(neighbour_pt1, neighbour_pt2,
                                                          inclusive=True), 'target_range'] = 1
        single_regress.local_df.loc[
            single_regress.local_df['prediction'].between(neighbour_pt2, neighbour_pt3,
                                                          inclusive=True), 'target_range'] = 2
        single_regress.local_df.loc[
            single_regress.local_df['prediction'].between(neighbour_pt3, neighbour_pt4,
                                                          inclusive=True), 'target_range'] = 3
        distances=neighbourhood_distances(explainer,single_regress)
        single_regress.local_df['distances'] = distances
        single_regress.local_df = single_regress.local_df.sort_values(by=['distances'])
        single_regress.local_df = single_regress.local_df.reset_index(drop=True)
        num_rows = CLEAR_settings.regression_sample_size * ((neighbour_pt2 -neighbour_pt1)/ (neighbour_pt4 - neighbour_pt1))
        temp_df = single_regress.local_df[single_regress.local_df['target_range'] == 1]
        temp_df = temp_df.sort_values(by=['distances'])
        temp_df = temp_df.iloc[0:int(num_rows), :]
        single_regress.neighbour_df = temp_df.copy(deep=True)
        num_rows = int(CLEAR_settings.regression_sample_size * (neighbour_pt3 - neighbour_pt2) / (
                neighbour_pt4 - neighbour_pt1))
        temp_df = single_regress.local_df[single_regress.local_df['target_range'] == 2]
        temp_df = temp_df.sort_values(by=['distances'])
        temp_df = temp_df.iloc[0:int(num_rows), :]
        single_regress.neighbour_df = single_regress.neighbour_df.append(temp_df, ignore_index=True)
        num_rows = int(CLEAR_settings.regression_sample_size * (neighbour_pt4 - neighbour_pt3) / (
                neighbour_pt4 - neighbour_pt1))
        temp_df = single_regress.local_df[single_regress.local_df['target_range'] == 3]
        temp_df = temp_df.sort_values(by=['distances'])
        temp_df = temp_df.iloc[0:int(num_rows), :]
        single_regress.neighbour_df = single_regress.neighbour_df.append(temp_df, ignore_index=True)
        single_regress.neighbour_df = single_regress.neighbour_df.sort_values(by=['distances'])
        single_regress.neighbour_df = single_regress.neighbour_df.reset_index(drop=True)
        if CLEAR_settings.generate_regression_files == True:
            filename1 = CLEAR_settings.CLEAR_path + 'local_' + str(
                single_regress.observation_num) + '_' + datetime.now().strftime("%Y%m%d-%H%M%S%f") + '.csv'
            single_regress.neighbour_df.to_csv(filename1)
    # Selects s observations of synthetic data based solely on distances
    elif CLEAR_settings.neighbourhood_algorithm == 'Unbalanced':
        distances=neighbourhood_distances(explainer,single_regress)
        single_regress.local_df['distances'] = distances
        single_regress.local_df = single_regress.local_df.sort_values(by=['distances'])
        single_regress.local_df = single_regress.local_df.reset_index(drop=True)
        temp_df = single_regress.local_df.iloc[0:int(200), :]
        single_regress.neighbour_df = temp_df.copy(deep=True)
        if CLEAR_settings.generate_regression_files == True:
            filename1 = CLEAR_settings.CLEAR_path + 'local_' + str(
                single_regress.observation_num) + '_' + datetime.now().strftime("%Y%m%d-%H%M%S%f") + '.csv'
            single_regress.neighbour_df.to_csv(filename1)
    else:
        print('Neighbourhood Algorithm Misspecified')
    return single_regress


def neighbourhood_distances(explainer,single_regress):
    x = single_regress.local_df.loc[0, explainer.cat_features].values.reshape(1, -1)
    y = single_regress.local_df.loc[:, explainer.cat_features].values.reshape(
        single_regress.local_df.shape[0], -1)
    z = cdist(x, y, 'jaccard')
    x = single_regress.local_df.loc[0, explainer.numeric_features].values.reshape(1, -1)
    y = single_regress.local_df.loc[:, explainer.numeric_features].values.reshape(
        single_regress.local_df.shape[0], -1)
    w = cdist(x, y, 'euclidean')
    distances = (len(explainer.numeric_features) / explainer.num_features * w + \
                 len(explainer.cat_features) / explainer.num_features * z).ravel()
    return distances


def get_counterfactuals(explainer, boundary_df, X_test_sample, multi_index):
    # for every_obs
    temp = copy.deepcopy(explainer.feature_list)
    temp.insert(0, 'observation')
    temp.append('prediction')
    explainer.counterf_rows_df = pd.DataFrame(columns=temp)
    for i in range(CLEAR_settings.first_obs, CLEAR_settings.last_obs + 1):
        data_row = pd.DataFrame(columns=explainer.feature_list)
        data_row = data_row.append(X_test_sample.iloc[i], ignore_index=True)
        data_row.fillna(0, inplace=True)
        y = forecast_data_row(explainer, data_row, multi_index)
        if CLEAR_settings.case_study not in ['IRIS', 'Glass']:
            for feature in explainer.numeric_features:
                temp_df = explainer.sensit_df[
                    (explainer.sensit_df['observation'] == i) & (
                            explainer.sensit_df['feature'] == feature)]
                temp_df = temp_df['probability'].agg(['min', 'max'])
                if (temp_df['min'] <= CLEAR_settings.binary_decision_boundary) & \
                        (temp_df['max'] > CLEAR_settings.binary_decision_boundary):
                    old_value = X_test_sample.loc[i, feature]
                    boundary = numeric_counterfactual(explainer, feature, old_value, i)
                    # This is necessary for cases where the observation in sensit_df nearest to 50th percentile
                    # is the last observation and hence is not identified by Get_Counterfactual.
                    if np.isnan(boundary):
                        continue
                    # estimate new_value corresponding to 50th percentile. This Gridsearch assumes only a single 50th percentile
                    s1 = X_test_sample.iloc[i].copy(deep=True)
                    s1['observation'] = i
                    s1['feature'] = feature
                    s1['target_range'] = 'counterf'
                    s1['distances'] = np.nan
                    s1.loc[feature] = boundary
                    explainer.counterf_rows_df = explainer.counterf_rows_df.append(s1, ignore_index=True)
            if y >= CLEAR_settings.binary_decision_boundary:
                current_class = 1
            else:
                current_class = 0
            #LIME comparison functionality is restricted to numeric features as this was the scope
            #of the analysis in the ECAI paper
            if CLEAR_settings.LIME_comparison is False:
                for feature in explainer.cat_features:
                    if X_test_sample.loc[i, feature] == 1:
                        continue
                    temp_df = explainer.catSensit_df[
                        (explainer.catSensit_df['observation'] == i) & (
                                explainer.catSensit_df['feature'] == feature)]
                    if temp_df.empty:
                        continue
                    temp_df = temp_df.copy(deep=True)
                    if temp_df.newnn_class[temp_df.index[0]] != current_class:
                        cat_idx = [X_test_sample.columns.get_loc(col) for col in X_test_sample if
                                   col.startswith(feature[:3])]
                        s1 = X_test_sample.iloc[i].copy(deep=True)
                        s1[cat_idx] = 0
                        s1.loc[feature] = 1
                        s1['feature'] = feature
                        s1['target_range'] = 'counterf'
                        s1['distances'] = np.nan
                        s1['observation'] = i
                        explainer.counterf_rows_df = explainer.counterf_rows_df.append(s1, ignore_index=True)

            if not explainer.counterf_rows_df.empty:
                explainer.num_counterf = explainer.counterf_rows_df.shape[0]
                CLEAR_pred_input_func = tf.estimator.inputs.pandas_input_fn(
                    x=explainer.counterf_rows_df.iloc[:, 1:-4],
                    batch_size=1,
                    num_epochs=1,
                    shuffle=False)
                predictions = explainer.model.predict(CLEAR_pred_input_func)
                y = np.array([])
                for p in predictions:
                    y = np.append(y, p['probabilities'][1])
                explainer.counterf_rows_df['prediction'] = y


        else:
            #for feature in X_test_sample.columns[0:-3]:
            for feature in X_test_sample.columns:
                if boundary_df.loc[i, feature + '_prob'] != 10000:
                    s1 = X_test_sample.iloc[i].copy(deep=True)
                    s2 = pd.Series(s1)
                    s2['feature'] = feature
                    s2['target_range'] = 'counterf'
                    s2['distances'] = np.nan
                    s2['observation'] = i
                    s2.loc[feature] = boundary_df.loc[i, feature + '_val']
                    explainer.counterf_rows_df = explainer.counterf_rows_df.append(s2, ignore_index=True)
            explainer.num_counterf = explainer.counterf_rows_df.shape[0]
            if not explainer.counterf_rows_df.empty:
                predictions = explainer.model.predict_proba(explainer.counterf_rows_df.iloc[:, 1:-4].values)
                explainer.counterf_rows_df['prediction'] = predictions[:, multi_index]
    return explainer


def numeric_counterfactual(explainer, feature, old_value, observation):
    temp_df = explainer.sensit_df.loc[
        (explainer.sensit_df['observation'] == observation) & (explainer.sensit_df['feature'] == feature)]
    # & (t_df['new_value']==t_df['new_value'].min())]
    temp_df.reset_index(inplace=True, drop=True)
    # get index of the data point nearest old value  i.e. where new_value - old_value = 0 (approx)
    m = temp_df.iloc[(temp_df['new_value'] - old_value).abs().argsort()[:1]].index[0]
    # get indexes of points minimum
    n = abs(temp_df['probability'] - CLEAR_settings.binary_decision_boundary).values
    minimums = list(argrelextrema(n, np.less)[0])
    # also find plateaus
    plateaus = list(np.where(np.diff(n) == 0)[0])
    if plateaus:
        plateaus.append(plateaus[-1] + 1)
    minimums = minimums + plateaus
    min_found = False
    while (len(minimums) != 0) and (min_found == False):
        nearest_min = minimums[abs(minimums - m).argmin()]
        # check if min corresponds to CLEAR_settings.binary_decision_boundary:
        if temp_df.loc[nearest_min, 'probability'] >= CLEAR_settings.binary_decision_boundary:
            if (temp_df.loc[nearest_min + 1, 'probability'] <= CLEAR_settings.binary_decision_boundary) \
                    or (temp_df.loc[nearest_min - 1, 'probability'] <= CLEAR_settings.binary_decision_boundary):
                min_found = True
            else:
                minimums.remove(nearest_min)
        elif temp_df.loc[nearest_min, 'probability'] < CLEAR_settings.binary_decision_boundary:
            if (temp_df.loc[nearest_min + 1, 'probability'] > CLEAR_settings.binary_decision_boundary) \
                    or (temp_df.loc[nearest_min - 1, 'probability'] > CLEAR_settings.binary_decision_boundary):
                min_found = True
            else:
                minimums.remove(nearest_min)
        else:
            print('error in probability data')

    if min_found == False:
        perc_counterf = np.NaN
    elif temp_df.loc[nearest_min, 'probability'] == CLEAR_settings.binary_decision_boundary:
        perc_counterf = temp_df.loc[nearest_min, 'new_value']
    else:

        a = temp_df.loc[nearest_min, 'probability']
        b = temp_df.loc[nearest_min, 'new_value']
        if temp_df.loc[nearest_min, 'probability'] >= CLEAR_settings.binary_decision_boundary:
            if temp_df.loc[nearest_min + 1, 'probability'] < CLEAR_settings.binary_decision_boundary:
                c = temp_df.loc[nearest_min + 1, 'probability']
                d = temp_df.loc[nearest_min + 1, 'new_value']
            else:
                c = temp_df.loc[nearest_min - 1, 'probability']
                d = temp_df.loc[nearest_min - 1, 'new_value']
        # a= nearest higer probability to binary_decision_boundary, and b = corresponding 'new_value' for the feature
        # c= nearest lower probability to binary_decision_boundary and d = corresponding 'new_value' for the feature
        # then 50th percentile = (b-d)*((binary_decision_boundary-c)/(a+c))+d
        elif temp_df.loc[nearest_min, 'probability'] < CLEAR_settings.binary_decision_boundary:
            if temp_df.loc[nearest_min + 1, 'probability'] > CLEAR_settings.binary_decision_boundary:
                c = temp_df.loc[nearest_min + 1, 'probability']
                d = temp_df.loc[nearest_min + 1, 'new_value']
            else:
                c = temp_df.loc[nearest_min - 1, 'probability']
                d = temp_df.loc[nearest_min - 1, 'new_value']
        perc_counterf = (b - d) * ((CLEAR_settings.binary_decision_boundary - c) / (a - c)) + d
    return (perc_counterf)


def perform_regression(explainer, single_regress):
    # A stepwise regression is performed. The user can specify a set of features
    # (‘selected’) that will be included in the regression. There are several
    # motivations for including this option: (i) it provides a means of ensuring that
    # the features from the dataset are included ‘unaltered’ i.e. not raised to a
    # power or included in an interaction term. This can lead to explanations that
    # are of greater interpretability, though perhaps of lower fidelity. For example,
    # with the IRIS dataset the user can specify that the regression equation should
    # include ['SepalL', 'SepalW', 'PetalL', 'PetalW'] i.e. the four features in the
    # dataset. CLEAR’s stepwise regression will then add further power/interaction
    # terms. (ii) it enables the user to use their domain knowledge to focus CLEAR’s
    # regressions (iii) it reduces running times.

    # In order to ensure that CLEAR’s regressions explain the c-counterfactuals it
    # identifies, CLEAR adds the dummy variables corresponding to each
    # c-counterfactual.

    # A future enhancement will be to carry out backward's regression.

    if CLEAR_settings.case_study == 'PIMA':
        selected = ['1', 'BloodP', 'Skin', 'BMI', 'Pregnancy', 'Glucose', 'Insulin', 'DiabF', 'Age']
    elif CLEAR_settings.case_study == 'IRIS':
        selected = ['1', 'SepalL', 'SepalW', 'PetalL', 'PetalW']
    elif CLEAR_settings.case_study == 'Glass':
        selected = ['1', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
    elif CLEAR_settings.case_study == 'BreastC':
        selected = ['1']
    elif explainer.cat_features != []:


        # for each categorical, if c-counterfactual exists include categorical for data_row plus for c-counterfactual
        # Also ensure that dummy trap is avoided
        dummy_trap = True
        counterfactualDummies = getCounterfactualDummies(explainer, single_regress.nn_forecast, \
                                                         single_regress.data_row, single_regress.observation_num, \
                                                         dummy_trap)
        if CLEAR_settings.case_study == 'Census':
            selected = ['1', 'age', 'hoursPerWeek']
        elif CLEAR_settings.case_study == 'Credit Card':
            selected = ['1', 'LIMITBAL', 'AGE', 'PAY0', 'PAY6', 'BILLAMT1', 'BILLAMT6', 'PAYAMT1', 'PAYAMT6']
        selected = selected + counterfactualDummies

    # Create poly_df excluding any categorical features with low sum
    X = single_regress.neighbour_df.iloc[:, 0:explainer.num_features].copy(deep=True)
    X = X.reset_index(drop=True)
    temp = [col for col, val in X.sum().iteritems() \
            if ((val <= CLEAR_settings.counterfactual_weight) and (col in explainer.cat_features) and
                col not in selected)]
    X.drop(temp, axis=1, inplace=True)
    temp = [col for col, val in X.sum().iteritems() \
            if ((val == X.shape[0]) and (col in explainer.cat_features) and
                col not in selected)]
    X.drop(temp, axis=1, inplace=True)


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

    poly_df_org_first_row = poly_df.iloc[0, :] + 0  # plus 0 is to get rid of 'negative zeros' ie python format bug
    org_poly_df = poly_df.copy(deep=True)

    # remove irrelevant categorical features from 'remaining' list of candidate independent features for regression
    remaining = poly_df.columns.tolist()
    temp = []
    for x in remaining:
        if x in selected:
            temp.append(x)
        elif (x[:3] in explainer.category_prefix) and (x.endswith('_sqrd')):
            temp.append(x)
        elif (x[:3] in explainer.category_prefix) and ('_' in x):
            if x[:3] == x.split("_", 1)[1][:3]:
                temp.append(x)
    remaining = [x for x in remaining if x not in temp]

    # If required, transform Y so that regression goes through the data point to be explained
    if CLEAR_settings.regression_type == 'multiple' and CLEAR_settings.centering is True:
        Y = single_regress.neighbour_df.loc[:, 'prediction'] - single_regress.nn_forecast
        poly_df = poly_df - poly_df.iloc[0, :]
    else:
        Y = single_regress.neighbour_df.loc[:, 'prediction'].copy(deep=True)

    Y = Y.reset_index(drop=True)
    Y_cont = Y.copy(deep=True)

    poly_df['prediction'] = pd.to_numeric(Y_cont, errors='coerce')
    current_score, best_new_score = -1000, -1000
    while remaining and current_score == best_new_score and len(selected) < CLEAR_settings.max_predictors:
        scores_with_candidates = []
        for candidate in remaining:
            if CLEAR_settings.regression_type == 'multiple' and CLEAR_settings.centering is True:
                formula = "{} ~ {}".format('prediction', ' + '.join(selected + [candidate]) + '-1')
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
                        score=sm.OLS.from_formula(formula, poly_df).fit(disp=0).rsquared_adj
                        if candidate== 'hoursPerWeek_sqrd':
                            qaz=1
                    else:
                        print("Error Ajusted R-squared is not used with logistic regression")

                else:
                    print('score type not correctly specified')
                    exit
                scores_with_candidates.append((score, candidate))
                del formula
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
    if CLEAR_settings.regression_type == 'multiple' and CLEAR_settings.centering is True:
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
            single_regress.prediction_score = classifier.aic
        elif CLEAR_settings.score_type == 'prsquared':
            single_regress.prediction_score = classifier.prsquared
        elif CLEAR_settings.score_type == 'adjR':
            single_regress.prediction_score = classifier.rsquared_adj
        else:
            print('incorrect score type')
        predictions = classifier.predict(poly_df)
        single_regress.features = selected
        single_regress.coeffs = classifier.params.values

        # This needs changing to allow for multi-class
        if CLEAR_settings.multi_class is False:
            if CLEAR_settings.regression_type == 'logistic':
                single_regress.accuracy = (classifier.pred_table()[0][0]
                                           + classifier.pred_table()[1][1]) / classifier.pred_table().sum()
            else:
                Z = Y.copy(deep=True)
                Z[Z >= CLEAR_settings.binary_decision_boundary] = 2
                Z[Z < CLEAR_settings.binary_decision_boundary] = 1
                Z[Z == 2] = 0
                W = predictions.copy(deep=True)
                W[W >= CLEAR_settings.binary_decision_boundary] = 2
                W[W < CLEAR_settings.binary_decision_boundary] = 1
                W[W == 2] = 0
                single_regress.accuracy = (W == Z).sum() / Z.shape[0]
        else:
            # Code for accuracy of multi-class classifiers has not yet been completed
            single_regress.accuracy = 'Not calculated'
        # Below code for multiclass currently assumes decision boundary at 0.5.


        # Calculate regression intercept and order regression coefficients for 'explainer_spreadsheet_data' (which is
        # written to CLRreg csv file)
        if CLEAR_settings.regression_type == 'logistic' or \
                (CLEAR_settings.regression_type == 'multiple' and CLEAR_settings.centering is False):
            single_regress.intercept = classifier.params[0]
            single_regress.spreadsheet_data = []
            for i in range(len(selected)):
                selected_feature = selected[i]
                for j in range(len(classifier.params)):
                    coeff_feature = classifier.params.index[j]
                    if selected_feature == coeff_feature:
                        single_regress.spreadsheet_data.append(poly_df_org_first_row.loc[selected_feature])
            single_regress.untransformed_predictions = classifier.predict(org_poly_df)
        else:
            single_regress.spreadsheet_data = []
            temp = 0
            single_regress.intercept = +single_regress.nn_forecast
            for i in range(len(selected)):
                selected_feature = selected[i]
                for j in range(len(classifier.params)):
                    coeff_feature = classifier.params.index[j]
                    if selected_feature == coeff_feature:
                        single_regress.intercept -= poly_df_org_first_row.loc[selected_feature] * classifier.params[
                            j]
                        temp -= poly_df_org_first_row.loc[selected_feature] * classifier.params[j]
                        single_regress.spreadsheet_data.append(poly_df_org_first_row.loc[selected_feature])
            adjustment = single_regress.nn_forecast - classifier.predict(poly_df_org_first_row)
            single_regress.untransformed_predictions = adjustment[0] + classifier.predict(org_poly_df)
    except:
        print(formula)
    #                input("Regression failed. Press Enter to continue...")
    # local prob is for the target point is in class 0 . CONFIRM!
    single_regress.local_prob = single_regress.untransformed_predictions[0]
    if CLEAR_settings.multi_class is True:
        single_regress.regression_class = ""  # identification of regression class requires a seperate regression for each multi class
    else:
        if single_regress.local_prob >= CLEAR_settings.binary_decision_boundary:
            single_regress.regression_class = 1
        else:
            single_regress.regression_class = 0
        if single_regress.nn_forecast >= CLEAR_settings.binary_decision_boundary:
            single_regress.nn_class = 1
        else:
            single_regress.nn_class = 0




    return single_regress


def getCounterfactualDummies(explainer, nn_forecast, data_row, observation_num,dummy_trap):
    if nn_forecast >= CLEAR_settings.binary_decision_boundary:
        current_class = 1
    else:
        current_class = 0
    temp_df = explainer.catSensit_df[
        (explainer.catSensit_df['observation'] == observation_num) &
        (explainer.catSensit_df.newnn_class != current_class)].copy(deep=True)
    # get categorical features which counterfactually change observation's class
    w = temp_df.feature.to_list()
    y = [x[:3] for x in w]
    y = list(set(y))
    z = [col for col in data_row if
         (col in explainer.cat_features) and (data_row.loc[0, col] == 1)]
    v = [x for x in z if x[:3] in y]
    # ensure that at least 1 dummy variable is excluded for each categorical feature
    if dummy_trap is True:
        for i in y:
            x1 = len([u for u in w if u.startswith(i)])  # no dummy variables selected from categorical sensitivity file
            x2 = len([u for u in explainer.cat_features if u.startswith(i)])  # dummy variables in data_row
            if x2 == x1 + 1:
                # drop dummy from w
                x3 = [u for u in w if u.startswith(i)][-1]
                w.remove(x3)
    return (w + v)
