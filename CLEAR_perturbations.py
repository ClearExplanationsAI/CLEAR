# Outstanding - Calculate_Perturbations() does not allow for categorical multi-class
#                                       create missing_log_df function
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from jinja2 import Environment, FileSystemLoader
from math import log10, floor, log, exp
from sympy import symbols, solve, simplify

import CLEAR_settings, CLEAR_regression


class CLEARPerturbation(object):
    # Contains features specific to a particular b-perturbation
    def __init__(self):
        self.wTx = None
        self.nncomp_idx = None
        self.multi_index = None
        self.target_feature = None
        self.obs = None
        self.newnn_class = None
        self.raw_weights = None
        self.raw_eqn = None
        self.raw_data = None
        self.adj_raw_data = None
        self.target_prob = None


def Calculate_Perturbations(explainer, results_df, multiClassBoundary_df, multi_index=None):
    """ b-perturbations are now calculated and stored
        in the nncomp_df dataframe. If CLEAR calculates a b-perturbation
        that is infeasible, then the details of the b-perturbation
        are stated in the missing_log_df dataframe. CLEAR will classify
        a b-perturbation as being infeasible if it is outside 'the feasibility
        range' it calculates for each feature.
    """
    print("\n Calculating b-counterfactuals \n")
    nncomp_df = pd.DataFrame(columns=['observation', 'multi_class', 'feature', 'orgFeatValue', 'orgAiProb',
                                      'actPerturbedFeatValue', 'AiProbWithActPerturbation', 'estPerturbedFeatValue',
                                      'errorPerturbation', 'regProbWithActPerturbation',
                                      'errorRegProbActPerturb', 'orgClass'])
    bPerturb = CLEARPerturbation()
    bPerturb.nncomp_idx = 1
    missing_log_df = pd.DataFrame(columns=['observation', 'feature', 'reason', 'perturbation'])
    first_obs = CLEAR_settings.first_obs
    last_obs = CLEAR_settings.last_obs + 1
    for i in range(first_obs, last_obs):
        s1 = pd.Series(results_df.local_data[i], explainer.feature_list)
        s2 = pd.DataFrame(columns=explainer.feature_list)
        s2 = s2.append(s1, ignore_index=True)
        x = symbols('x')
        bPerturb.raw_eqn = results_df.loc[i, 'features'].copy()
        bPerturb.raw_weights = results_df.loc[i, 'weights'].tolist()
        bPerturb.raw_data = results_df.loc[i, 'local_data'].tolist()
        features_processed = 0
        # the next 3 lines ensures that the same code can be used irrespective of whether the regression has
        # been forced through the data point to be explained
        if bPerturb.raw_eqn == '1':
            bPerturb.raw_eqn.remove('1')
            bPerturb.raw_weights.pop(0)
        for j in range(0, len(explainer.feature_list)):
            features_processed += 1
            bPerturb.target_feature_weight = 0
            bPerturb.target_feature = explainer.feature_list[j]
            old_value = s2.iloc[0, j]
            # set target probability for b-perturbation
            if len(explainer.class_labels)>2:
                bPerturb.target_prob = multiClassBoundary_df.loc[i, bPerturb.target_feature + '_prob']
                if bPerturb.target_prob == 10000:
                    continue
            else:
                bPerturb.target_prob = CLEAR_settings.binary_decision_boundary

            # establish if b-perturbation exists
            if bPerturb.target_feature in explainer.numeric_features:
                temp_df = explainer.sensit_df[(explainer.sensit_df['observation'] == i) & (
                        explainer.sensit_df['feature'] == bPerturb.target_feature)]
                temp_df = temp_df['probability'].agg(['min', 'max'])
                if not (temp_df['min'] <= bPerturb.target_prob) & (temp_df['max'] > bPerturb.target_prob):
                    continue
            elif bPerturb.target_feature in explainer.cat_features:
                # CLEAR only considers b-perturbations from target feature = 1, that change its class
                if old_value != 1:
                    continue
                temp = [x for x in explainer.cat_features if
                        ((x[:3] == bPerturb.target_feature[:3]) and (x != bPerturb.target_feature))]
                temp_df = explainer.catSensit_df[(explainer.catSensit_df['observation'] == i) & (
                    explainer.catSensit_df['feature'].isin(temp))]
                if bPerturb.target_prob >= results_df.loc[i, 'nn_forecast']:
                    if all(bPerturb.target_prob >= temp_df['probability']):
                        continue
                elif all(bPerturb.target_prob < temp_df['probability']):
                    continue

            # establish if feature is in equation
            if not any(bPerturb.target_feature in s for s in bPerturb.raw_eqn):
                if missing_log_df.empty:
                    idx = 0
                else:
                    idx = missing_log_df.index.max() + 1
                missing_log_df.loc[idx, 'observation'] = i
                missing_log_df.loc[idx, 'feature'] = bPerturb.target_feature
                missing_log_df.loc[idx, 'reason'] = 'not in raw equation'
                continue

            # If target feature is numeric then create equation string
            if bPerturb.target_feature in explainer.numeric_features:
                str_eqn, bPerturb.target_feature_weight = generateString(explainer, results_df, i, bPerturb)
                # Solve the equation and check if there is a solution with a 'feasible' value
                solution = []
                eqn_roots = solve(str_eqn, x)
                for k in eqn_roots:
                    if k.is_real:
                        solution.append(k)

                    elif k == eqn_roots[len(eqn_roots) - 1]:
                        if missing_log_df.empty:
                            idx = 0
                        else:
                            idx = missing_log_df.index.max() + 1
                        missing_log_df.loc[idx, 'feature'] = bPerturb.target_feature
                        missing_log_df.loc[idx, 'observation'] = i
                        missing_log_df.loc[idx, 'reason'] = 'value not real'
                        estPerturbedFeatValue = None
                        continue
                # get minimum perturbation
                if len(solution) > 0:
                    temp2 = []
                    for y in solution:
                        if explainer.feature_min[bPerturb.target_feature] <= y <= explainer.feature_max[
                            bPerturb.target_feature]:
                            temp2.append(y)
                    if len(temp2) > 0:
                        valid_roots = temp2 - old_value
                        estPerturbedFeatValue = min(valid_roots, key=abs)
                        estPerturbedFeatValue = estPerturbedFeatValue + old_value
                    else:
                        # if roots are all infeasible, take root nearest to feasibility range
                        lowest_root = 999
                        for y in solution:
                            k = min(abs(explainer.feature_min[bPerturb.target_feature] - y),
                                    abs(explainer.feature_max[bPerturb.target_feature] - y))
                            if k < lowest_root:
                                lowest_root = k
                                estPerturbedFeatValue = y
                        if lowest_root == 999:
                            continue  # i.e. go to next feature in bPerturb.raw_eqn

                else:
                    continue

                # update the observation to be explained (whose feature values are in s2)
                estPerturbedFeatValue = np.float64(estPerturbedFeatValue)
                s2.iloc[0, j] = estPerturbedFeatValue

                if multi_index is None:
                    temp = 'N\A'
                else:
                    temp = explainer.class_labels[multi_index]
                bPerturb.nncomp_idx += 1
                nncomp_df.loc[bPerturb.nncomp_idx, 'observation'] = i
                nncomp_df.loc[bPerturb.nncomp_idx, 'multi_class'] = temp
                nncomp_df.loc[bPerturb.nncomp_idx, 'feature'] = bPerturb.target_feature
                nncomp_df.loc[bPerturb.nncomp_idx, 'orgFeatValue'] = old_value
                nncomp_df.loc[bPerturb.nncomp_idx, 'orgAiProb'] = results_df.loc[i, 'nn_forecast']
                nncomp_df.loc[bPerturb.nncomp_idx, 'estPerturbedFeatValue'] = estPerturbedFeatValue
                AiProbWithActPerturbation = \
                explainer.counterf_rows_df.prediction[(explainer.counterf_rows_df['feature'] == \
                                                       bPerturb.target_feature) & (
                                                                  explainer.counterf_rows_df['observation'] == i)].iloc[
                    0]
                nncomp_df.loc[bPerturb.nncomp_idx, 'AiProbWithActPerturbation'] = AiProbWithActPerturbation
                nncomp_df.loc[bPerturb.nncomp_idx, 'orgClass'] = results_df.loc[i, 'regression_class']
                s2.iloc[0, j] = old_value

                # estimate estPerturbedFeatValue corresponding to the decision boundary
                if len(explainer.class_labels)>2:
                    boundary_val = multiClassBoundary_df.loc[i, bPerturb.target_feature + '_val']
                else:
                    boundary_val = CLEAR_regression.numeric_counterfactual(explainer, bPerturb.target_feature,
                                                                           old_value, i)
                nncomp_df.loc[bPerturb.nncomp_idx, 'actPerturbedFeatValue'] = boundary_val
                nncomp_df.loc[bPerturb.nncomp_idx, 'errorPerturbation'] = abs(estPerturbedFeatValue - boundary_val)
                str_eqn = str_eqn.replace('x', str(boundary_val))
                bPerturb.wTx = simplify(str_eqn)
                if CLEAR_settings.regression_type == 'multiple':
                    regProbWithActPerturbation = bPerturb.wTx + bPerturb.target_prob
                else:
                    regProbWithActPerturbation = 1 / (1 + exp(-bPerturb.wTx))
                nncomp_df.loc[bPerturb.nncomp_idx, 'regProbWithActPerturbation'] = regProbWithActPerturbation
                nncomp_df.loc[bPerturb.nncomp_idx, 'errorRegProbActPerturb'] = \
                    abs(regProbWithActPerturbation - AiProbWithActPerturbation)

            elif (bPerturb.target_feature in explainer.cat_features):
                # Create equation string
                obsData_df = pd.DataFrame(columns=explainer.feature_list)
                obsData_df.loc[0] = results_df.loc[i, 'local_data']
                dummy_trap = False
                counterfactualDummies = CLEAR_regression.getCounterfactualDummies(explainer,
                                                                                  results_df.loc[i, 'nn_forecast'],
                                                                                  obsData_df, i, dummy_trap)
                y = [x for x in counterfactualDummies if
                     (x.startswith(bPerturb.target_feature[:3]) and x != bPerturb.target_feature)]
                for k in y:
                    bPerturb.adj_raw_data = list(bPerturb.raw_data)
                    bPerturb.adj_raw_data[explainer.feature_list.index(k)] = 1
                    str_eqn, bPerturb.target_feature_weight = generateString(explainer, results_df, i, bPerturb)
                    str_eqn = str_eqn.replace('x', '0')
                    bPerturb.wTx = simplify(str_eqn)
                    nncomp_df = catUpdateNncomp_df(explainer, nncomp_df, bPerturb, multi_index, i, results_df, k)

    nncomp_df.observation = nncomp_df.observation.astype(int)
    nncomp_df.reset_index(inplace=True, drop=True)

    """
    Determines the actual values of the AI decision boundary for numeric features. This will then be used 
    for determining the fidelity errors of the CLEAR perturbations.
    """
    return nncomp_df, missing_log_df


def catUpdateNncomp_df(explainer, nncomp_df, bPerturb, multi_index, i, results_df, k):
    temp_df = explainer.catSensit_df[
        (explainer.catSensit_df['observation'] == i) &
        (explainer.catSensit_df.feature == k)]
    temp_df = temp_df.copy(deep=True)
    temp_df.reset_index(inplace=True, drop=True)
    AiProbWithActPerturbation = temp_df.loc[0, 'probability']
    if multi_index is None:
        temp = 'N\A'
    else:
        temp = CLEAR_settings.class_labels[multi_index]
    bPerturb.nncomp_idx += 1
    nncomp_df.loc[bPerturb.nncomp_idx, 'observation'] = i
    nncomp_df.loc[bPerturb.nncomp_idx, 'feature'] = bPerturb.target_feature
    nncomp_df.loc[bPerturb.nncomp_idx, 'multi_class'] = temp
    nncomp_df.loc[bPerturb.nncomp_idx, 'AiProbWithActPerturbation'] = np.float64(AiProbWithActPerturbation)
    nncomp_df.loc[bPerturb.nncomp_idx, 'orgAiProb'] = results_df.loc[i, 'nn_forecast']
    nncomp_df.loc[bPerturb.nncomp_idx, 'orgClass'] = results_df.loc[
        i, 'regression_class']  # needs correcting not sure if regression class needs reporting
    nncomp_df.loc[bPerturb.nncomp_idx, 'orgFeatValue'] = bPerturb.target_feature
    nncomp_df.loc[bPerturb.nncomp_idx, 'actPerturbedFeatValue'] = temp_df.loc[0, 'feature']
    if CLEAR_settings.regression_type == 'multiple':
        regProbWithActPerturbation = bPerturb.wTx
    else:
        regProbWithActPerturbation = 1 / (1 + exp(-bPerturb.wTx))
    nncomp_df.loc[bPerturb.nncomp_idx, 'regProbWithActPerturbation'] = np.float64(regProbWithActPerturbation)
    nncomp_df.loc[bPerturb.nncomp_idx, 'errorRegProbActPerturb'] = \
        round(abs(regProbWithActPerturbation - AiProbWithActPerturbation), 2)
    return (nncomp_df)


def generateString(explainer, results_df, observation, bPerturb):
    # For numeric target features the str eqn is used to calculate b-perturbations
    # For categorical target features str_eqn is used to calculate the c-counterfactuals
    if bPerturb.target_feature in explainer.numeric_features:
        raw_data = list(bPerturb.raw_data)
        if CLEAR_settings.regression_type == 'multiple':
            str_eqn = '-' + str(bPerturb.target_prob) + '+' + str(results_df.loc[observation, 'intercept'])
        else:  # ie logistic regression
            # bPerturb.wTx = -ln((1-p)/p)
            temp = -log((1 - bPerturb.target_prob) / bPerturb.target_prob)
            str_eqn = str(-temp) + '+' + str(results_df.loc[observation, 'intercept'])

    else:
        raw_data = list(bPerturb.adj_raw_data)
        str_eqn = '+' + str(results_df.loc[observation, 'intercept'])

    for raw_feature in bPerturb.raw_eqn:
        if raw_feature == '1':
            pass
        elif raw_feature == bPerturb.target_feature:
            str_eqn += "+" + str(bPerturb.raw_weights[bPerturb.raw_eqn.index(raw_feature)]) + "*x"
            bPerturb.target_feature_weight = bPerturb.raw_weights[bPerturb.raw_eqn.index(raw_feature)]
        elif raw_feature in explainer.feature_list:
            new_term = raw_data[explainer.feature_list.index(raw_feature)] * bPerturb.raw_weights[
                bPerturb.raw_eqn.index(raw_feature)]
            str_eqn += "+ " + str(new_term)
        elif raw_feature == str(bPerturb.target_feature) + "_sqrd":
            str_eqn += "+" + str(bPerturb.raw_weights[bPerturb.raw_eqn.index(raw_feature)]) + "*x**2"
        elif raw_feature.endswith('_sqrd'):
            new_term = raw_feature.replace('_sqrd', '')
            new_term = (raw_data[explainer.feature_list.index(new_term)] ** 2) * bPerturb.raw_weights[
                bPerturb.raw_eqn.index(raw_feature)]
            str_eqn += "+ " + str(new_term)
        elif '_' in raw_feature:
            interaction_terms = raw_feature.split('_')
            if interaction_terms[0] == bPerturb.target_feature:
                new_term = str(raw_data[explainer.feature_list.index(interaction_terms[1])] \
                               * bPerturb.raw_weights[bPerturb.raw_eqn.index(raw_feature)]) + "*x"
            elif interaction_terms[1] == bPerturb.target_feature:
                new_term = str(raw_data[explainer.feature_list.index(interaction_terms[0])] \
                               * bPerturb.raw_weights[bPerturb.raw_eqn.index(raw_feature)]) + "*x"
            else:
                new_term = str(raw_data[explainer.feature_list.index(interaction_terms[0])]
                               * raw_data[explainer.feature_list.index(interaction_terms[1])]
                               * bPerturb.raw_weights[bPerturb.raw_eqn.index(raw_feature)])
            str_eqn += "+ " + new_term
        else:
            print("error in processing equation string")
        pass
    return str_eqn, bPerturb.target_feature_weight


def Summary_stats(nncomp_df, missing_log_df):
    """ Create summary statistics and frequency histogram
    """
    if nncomp_df.empty:
        print('no data for plot')
        return
    less_target_sd = 0
    temp_df = nncomp_df.copy(deep=True)
    temp_df = temp_df[~temp_df.errorPerturbation.isna()]
    if temp_df['errorPerturbation'].count() != 0:
        less_target_sd = temp_df[temp_df.errorPerturbation <= 0.25].errorPerturbation.count()
        x = temp_df['errorPerturbation']
        x = x[~x.isna()]
        ax = x.plot.hist(grid=True, bins=20, rwidth=0.9)
        plt.title(
            'perturbations = ' + str(temp_df['errorPerturbation'].count()) + '  Freq Counts <= 0.25 sd = ' + str(
                less_target_sd)
            + '\n' + 'regression = ' + CLEAR_settings.regression_type + ', score = ' + CLEAR_settings.score_type
            + ', sample = ' + str(CLEAR_settings.num_samples)
            + '\n' + 'max_predictors = ' + str(CLEAR_settings.max_predictors)
            + ', regression_sample_size = ' + str(CLEAR_settings.regression_sample_size))
        plt.xlabel('Standard Deviations')
        fig = ax.get_figure()
        fig.savefig(CLEAR_settings.CLEAR_path + 'hist' + datetime.now().strftime("%Y%m%d-%H%M") + '.png',
                    bbox_inches="tight")
    else:
        print('no data for plot')
        temp_df = nncomp_df.copy(deep=True)
    # x=np.array(nncomp_df['errorPerturbation'])

    filename1 = CLEAR_settings.CLEAR_path + 'wPerturb_' + datetime.now().strftime("%Y%m%d-%H%M") + '.csv'
    nncomp_df.to_csv(filename1)
    filename2 = CLEAR_settings.CLEAR_path + 'missing_' + datetime.now().strftime("%Y%m%d-%H%M") + '.csv'
    missing_log_df.to_csv(filename2)
    output = [CLEAR_settings.sample_model, less_target_sd]
    filename3 = 'batch.csv'
    try:
        with open(CLEAR_settings.CLEAR_path + filename3, 'a') as file1:
            writes = csv.writer(file1, delimiter=',', skipinitialspace=True)
            writes.writerow(output)
        file1.close()
    except:
        pass
    return


def Single_prediction_report(results_df, nncomp_df, single_regress, explainer):
    if nncomp_df.empty:
        print('no data for single prediction report')
        return
    if len(explainer.class_labels)==2:
        explanandum= explainer.class_labels[1]
    else:
        explanandum = CLEAR_settings.multi_class_focus


    def round_sig(x, sig=2):
        if type(x) == np.ndarray:
            x = x[0]
        if x == 0:
            y= 0
        else:
            y= round(x, sig - int(floor(log10(abs(x)))) - 1)
        return y

    j = results_df.index.values[0]
    if CLEAR_settings.regression_type == 'multiple':
        regression_formula = 'prediction = ' + str(round_sig(results_df.intercept[j]))
    else:
        regression_formula = '<font size = "4.5">prediction =  [ 1 + e<sup><b>-w<sup>T</sup>x</sup></b> ]<sup> -1</sup></font size><br><br>' \
                             + '<font size = "4.5"><b><i>w</i></b><sup>T</sup><b><i>x</font size></i></b> =  ' + str(
            round_sig(results_df.intercept[j]))

    for i in range(len(results_df.features[j])):
        if ("_" in results_df.features[j][i]) and ("_sqrd" not in results_df.features[j][i]):
            results_df.features[j][i] = "(" + results_df.features[j][i] + ")"
        for h in range(results_df.features[j][i].count("Dd")):
            t = results_df.features[j][i].find("Dd")
            if t == 3 and len(results_df.features[j][i]) > 6:
                results_df.features[j][i] = results_df.features[j][i][5:]
            elif t == 3 and len(results_df.features[j][i]) <= 6:
                results_df.features[j][i] = results_df.features[j][i][0:3] + results_df.features[j][i][t + 2:]
            elif len(results_df.features[j][i][t + 2:]) > 2:
                results_df.features[j][i] = results_df.features[j][i][:t - 3] + results_df.features[j][i][t + 2:]
            else:
                results_df.features[j][i] = results_df.features[j][i][:t] + results_df.features[j][i][t + 2:]
        if results_df.features[j][i] == '1':
            continue
        elif results_df.weights[j][i] < 0:
            regression_formula = regression_formula + ' - ' + str(-1 * round_sig(results_df.weights[j][i])) + \
                                 ' ' + results_df.features[j][i]
        else:
            regression_formula = regression_formula + ' + ' + str(round_sig(results_df.weights[j][i])) + \
                                 ' ' + results_df.features[j][i]
    regression_formula = regression_formula.replace("_sqrd", "<sup>2</sup>")
    regression_formula = regression_formula.replace("_", "*")
    report_AI_prediction = str(round_sig(results_df.nn_forecast[j]))
    if CLEAR_settings.score_type == 'adjR':
        regression_score_type = "Adjusted R-Squared"
    else:
        regression_score_type = CLEAR_settings.score_type
    # get rid of dummy variables equal to zero
    temp2_df = pd.DataFrame(columns=['Feature', 'Input Value'])
    temp = [col for col in single_regress.data_row.columns \
            if not ((single_regress.data_row.loc[0, col] == 0) and (col in explainer.cat_features))]
    input_data = single_regress.data_row.loc[0, temp]
    k = 0
    for col in input_data.index:
        if col in explainer.cat_features:
            temp2_df.loc[k, 'Feature'] = col.replace("Dd", "=")
            temp2_df.loc[k, 'Input Value'] = "1"
        else:
            temp2_df.loc[k, 'Feature'] = col
            temp2_df.loc[k, 'Input Value'] = str(round(input_data.iloc[k], 2))
        k += 1
    inputData_df = temp2_df.copy(deep=True)
    inputData_df.set_index('Feature', inplace=True)
    inputData_df = inputData_df.transpose().copy(deep=True)
    temp_df = nncomp_df.copy(deep=True)
    temp_df = temp_df[~temp_df.errorPerturbation.isna()]
    temp_df['errorPerturbation'] = temp_df['estPerturbedFeatValue'] - temp_df['actPerturbedFeatValue']

    counter_df = temp_df[['feature', 'orgFeatValue', 'actPerturbedFeatValue', 'estPerturbedFeatValue', \
                          'errorPerturbation']].copy()
    counter_df.errorPerturbation = abs(counter_df.errorPerturbation)
    counter_df.rename(columns={"orgFeatValue": "input value", "actPerturbedFeatValue": "actual b-counterfactual value", \
                               "estPerturbedFeatValue": "regression estimated b-counterfactual value",
                               "errorPerturbation": "b-counterfactual fidelity error"}, inplace=True)
    #    HTML_df.to_html('CLEAR.HTML')

    temp_df = nncomp_df.copy(deep=True)
    temp_df = temp_df[temp_df.feature.isin(explainer.cat_features)]
    temp_df.loc[temp_df.feature.str.contains('Dd'), 'feature'] = temp_df.feature.str[:3]
    temp_df.loc[temp_df.orgFeatValue.str.contains('Dd'), 'orgFeatValue'] = temp_df.orgFeatValue.str[5:]
    temp_df.loc[
        temp_df.actPerturbedFeatValue.str.contains('Dd'), 'actPerturbedFeatValue'] = temp_df.actPerturbedFeatValue.str[
                                                                                     5:]

    c_counter_df = temp_df[['feature', 'orgFeatValue', 'actPerturbedFeatValue', 'AiProbWithActPerturbation',
                            'regProbWithActPerturbation', 'errorRegProbActPerturb']].copy()
    c_counter_df.rename(columns={"orgFeatValue": "input value", "actPerturbedFeatValue": "c-counterfactual value",
                                 "AiProbWithActPerturbation": "actual c-counterfactual value",
                                 "regProbWithActPerturbation": "regression forecast using c-counterfactual",
                                 "errorRegProbActPerturb": "regression forecast error"},
                        inplace=True)

    # sorted unique feature list for the 'select features' checkbox
    feature_box = results_df.features[j]
    feature_box = ",".join(feature_box).replace('(', '').replace(')', '').replace('_', ',').split(",")
    feature_box = sorted(list(set(feature_box)), key=str.lower)
    for x in ['sqrd', '1']:
        if x in feature_box:
            feature_box.remove(x)
    # results_df.weights needs pre-processing prior to sending to HTML
    weights = results_df.weights.values[0]
    weights = weights.tolist()
    spreadsheet_data = results_df.spreadsheet_data.values[0]
    if len(weights) == len(spreadsheet_data) + 1:
        weights.pop(0)

    pd.set_option('colheader_justify', 'left', 'precision', 2)
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template("CLEAR_report.html")
    template_vars = {"title": "CLEAR Statistics",
                     "input_data_table": inputData_df.to_html(index=False, classes='mystyle'),
                     "counterfactual_table": counter_df.to_html(index=False, classes='mystyle'),
                     #                     "inaccurate_table": inaccurate_df.to_html(index=False,classes='mystyle'),
                     "dataset_name": CLEAR_settings.sample_model,
                     "explanadum": explanandum,
                     "observation_number": j,
                     "regression_formula": regression_formula,
                     "prediction_score": round_sig(results_df.Reg_Score[j]),
                     "regression_score_type": regression_score_type,
                     "regression_type": CLEAR_settings.regression_type,
                     "AI_prediction": report_AI_prediction,
                     "cat_counterfactual_table": c_counter_df.to_html(index=False, classes='mystyle'),
                     "feature_list": feature_box,
                     "spreadsheet_data": spreadsheet_data,
                     "weights": weights,
                     "intercept": results_df.intercept.values[0],
                     }
    with open('CLEAR_prediction_report.html', 'w') as fh:
        fh.write(template.render(template_vars))

    fig = plt.figure()
    plt.scatter(single_regress.neighbour_df.loc[:, 'prediction'], single_regress.untransformed_predictions, c='green',
                s=10)
    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), c="red", linestyle='-')

    plt.xlabel('Target AI System')
    if CLEAR_settings.regression_type == 'logistic':
        plt.ylabel('CLEAR Logistics Regression')
    elif CLEAR_settings.regression_type == 'multiple':
        plt.ylabel('CLEAR Multiple Regression')
    else:
        plt.ylabel('CLEAR Polynomial Regression')

    fig.savefig('CLEAR_plot.png', bbox_inches="tight")
    return ()
