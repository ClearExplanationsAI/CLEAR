from __future__ import print_function

import csv
import os

import numpy as np
import pandas as pd
import tensorflow as tf

import CLEAR_Process_Dataset
import CLEAR_settings

# I have not done multiclass categorical
def Create_sensitivity():
    np.random.seed(1)
    if CLEAR_settings.case_study == 'Credit Card':
        (X_train, X_test_sample, model, numeric_features, category_prefix,
         feature_list) = CLEAR_Process_Dataset.Create_Credit_Datasets()
    elif CLEAR_settings.case_study == 'PIMA':
        (X_train, X_test_sample, model, numeric_features, category_prefix,
         feature_list) = CLEAR_Process_Dataset.Create_PIMA_Datasets()
    elif CLEAR_settings.case_study == 'Census':
        (X_train, X_test_sample, model, numeric_features, category_prefix,
         feature_list) = CLEAR_Process_Dataset.Create_Census_Datasets()
    elif CLEAR_settings.case_study == 'BreastC':
        (X_train, X_test_sample, model, numeric_features, category_prefix,
         feature_list) = CLEAR_Process_Dataset.Create_BreastC_Datasets()
    elif CLEAR_settings.case_study in ['IRIS','Glass']:
        (X_train, X_test_sample, model, numeric_features, category_prefix,
         feature_list) = CLEAR_Process_Dataset.Create_Numeric_Multi_Datasets()

    if CLEAR_settings.use_prev_sensitivity is False:
        print('\n Performing grid search - step 1 of CLEAR method \n')
        feature_min = X_train.quantile(.01)
        feature_max = X_train.quantile(.99)
        categorical_features = [x for x in X_test_sample.columns if x[:3] in category_prefix]
        try:
            os.remove(CLEAR_settings.CLEAR_path + 'numericTemp.csv')
            os.remove(CLEAR_settings.CLEAR_path + 'categoricalTemp.csv')
        except OSError:
            pass
#create feature data for sensitivity files
        X_test_sample.reset_index(inplace=True, drop=True)

        for i in range(CLEAR_settings.first_obs, CLEAR_settings.last_obs + 1):

            for j in numeric_features:
                sensitivity_num = 250
                te = np.tile(X_test_sample.iloc[i, :].values, [sensitivity_num, 1])
                te_c = X_test_sample.columns.get_loc(j)
                te[:, te_c] = np.linspace(feature_min[j], feature_max[j], sensitivity_num)
                f = open(CLEAR_settings.CLEAR_path + 'numericTemp.csv', 'a')
                np.savetxt(f, te, delimiter=',')
                f.close()

        if len(category_prefix)!=0:
            for i in range(CLEAR_settings.first_obs, CLEAR_settings.last_obs + 1):
                for j in category_prefix:
                    cat_idx = [X_test_sample.columns.get_loc(col) for col in X_test_sample if col.startswith(j)]
                    if len(cat_idx) < 2:
                        continue
                    cat_num = len(cat_idx)
                    te = np.tile(X_test_sample.iloc[i, :].values, [cat_num, 1])
                    k = 0
                    for m in cat_idx:
                        te[k][cat_idx] = 0
                        te[k][m] = 1
                        k += 1
                    f = open(CLEAR_settings.CLEAR_path + 'categoricalTemp.csv', 'a')
                    np.savetxt(f, te, delimiter=',')
                    f.close()

#create one sensitivity file if dataset is binary, otherwise one sensitivity file per multi-class
        sensit_df = pd.read_csv(CLEAR_settings.CLEAR_path + 'numericTemp.csv', header=None, names=feature_list)

        if CLEAR_settings.multi_class is True:
            if CLEAR_settings.multi_class_focus == 'All':
                num_class = len(CLEAR_settings.multi_class_labels)
                multi_index = 0
            else:
                num_class = 1
                multi_index = CLEAR_settings.multi_class_labels.index(CLEAR_settings.multi_class_focus)
            for c in range(num_class):
                predictions = model.predict_proba(sensit_df.values)
                temp_df = pd.DataFrame(columns=['observation', 'feature', 'newnn_class', 'probability', 'new_value'])
                cnt = 0
                for i in range(CLEAR_settings.first_obs, CLEAR_settings.last_obs + 1):
                    for j in numeric_features:
                        for k in range(sensitivity_num):
                            temp_df.loc[cnt, 'observation'] = i
                            temp_df.loc[cnt, 'feature'] = feature_list[(cnt // sensitivity_num) % len(numeric_features)]
                            temp_df.loc[cnt, 'newnn_class'] = np.argmax(predictions[cnt])
                            temp_df.loc[cnt, 'probability'] = predictions[cnt,multi_index]
                            temp_df.loc[cnt, 'new_value'] = sensit_df.loc[cnt, j]
                            cnt += 1
                sensitivity_file = CLEAR_settings.case_study + '_sensitivity_m'+ str(multi_index) +'_' + \
                                   str(CLEAR_settings.test_sample) + '.csv'
                filename1 = CLEAR_settings.CLEAR_path + sensitivity_file
                temp_df.to_csv(filename1,index=False)
                multi_index +=1

        elif CLEAR_settings.use_sklearn == True:
                predictions = model.predict_proba(sensit_df.values)
                temp_df = pd.DataFrame(
                    columns=['observation', 'feature', 'newnn_class', 'probability', 'new_value'])
                cnt = 0
                for i in range(CLEAR_settings.first_obs, CLEAR_settings.last_obs + 1):
                    for j in numeric_features:
                        for k in range(sensitivity_num):
                            temp_df.loc[cnt, 'observation'] = i
                            temp_df.loc[cnt, 'feature'] = feature_list[
                                (cnt // sensitivity_num) % len(numeric_features)]
                            temp_df.loc[cnt, 'newnn_class'] = np.argmax(predictions[cnt])
                            temp_df.loc[cnt, 'probability'] = predictions[cnt, 1]
                            temp_df.loc[cnt, 'new_value'] = sensit_df.loc[cnt, j]
                            cnt += 1
                sensitivity_file = CLEAR_settings.case_study + '_sensitivity_' + str(CLEAR_settings.test_sample) + '.csv'
                filename1 = CLEAR_settings.CLEAR_path + sensitivity_file
                temp_df.to_csv(filename1)

       # TensorFlow model
        else:
       # First numeric features
            CLEAR_pred_input_func = tf.estimator.inputs.pandas_input_fn(
                x=sensit_df,
                batch_size=5000,
                num_epochs=1,
                shuffle=False)
            predictions = model.predict(CLEAR_pred_input_func)

            sensitivity_file = CLEAR_settings.case_study + '_sensitivity_' + str(CLEAR_settings.test_sample) + '.csv'
            init_cnt = sensitivity_num *CLEAR_settings.first_obs*len(numeric_features)
            cnt = 0
            top_row = ['observation', 'feature', 'newnn_class', 'probability', 'new_value']
            temp = len(numeric_features)

            with open(CLEAR_settings.CLEAR_path + sensitivity_file, 'w',newline='') as file1:
                writes = csv.writer(file1, delimiter=',', skipinitialspace=True)
                writes.writerow(top_row)
                try:
                    for p in predictions:
                        feature = numeric_features[((cnt // sensitivity_num)) % temp]
                        observation = (cnt + init_cnt) // (sensitivity_num * temp)
                        #                        observation = cnt // (sensitivity_num * len(numeric_features)) + 1
                        out_list = [observation, feature, p['class_ids'][0], p['probabilities'][1],
                                    sensit_df.loc[cnt, feature]]
                        cnt += 1
                        writes.writerow(out_list)
                except:
                    temp9 = 1
            file1.close()
        # then categorical features

            if len(category_prefix)!=0:
                catSensit_df = pd.read_csv(CLEAR_settings.CLEAR_path + 'categoricalTemp.csv', header=None, names=feature_list)
                CLEAR_pred_input_func = tf.estimator.inputs.pandas_input_fn(
                    x=catSensit_df,
                    batch_size=5000,
                    num_epochs=1,
                    shuffle=False)
                predictions = model.predict(CLEAR_pred_input_func)


                sensitivity_file = CLEAR_settings.case_study + '_catSensitivity_' + str(CLEAR_settings.test_sample) + '.csv'
                init_cnt = CLEAR_settings.first_obs * len(categorical_features)
                cnt = 0
                top_row = ['observation', 'feature', 'newnn_class', 'probability', 'new_value']
                temp = len(categorical_features)

                with open(CLEAR_settings.CLEAR_path + sensitivity_file, 'w', newline='') as file1:
                    writes = csv.writer(file1, delimiter=',', skipinitialspace=True)
                    writes.writerow(top_row)
                    try:
                        for p in predictions:
                            feature = categorical_features[(cnt % temp)]
                            observation = (cnt + init_cnt) //  temp
                            #                        observation = cnt // (sensitivity_num * len(numeric_features)) + 1
                            out_list = [observation, feature, p['class_ids'][0], p['probabilities'][1],
                                        catSensit_df.loc[cnt, feature]]
                            cnt += 1
                            writes.writerow(out_list)
                    except:
                        temp9 = 1
                file1.close()


    # only a small proportion of observations have feasible w-perturbations

    return X_train, X_test_sample, model, numeric_features, category_prefix, feature_list

