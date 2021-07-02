from __future__ import print_function

import csv
import os
import numpy as np
import pandas as pd
import CLEAR_settings


def Create_sensitivity(X_train, X_test_sample, model):
    print('\n Performing grid search - step 1 of CLEAR method \n')
    feature_list = X_train.columns.tolist()
    feature_min = X_train.quantile(.01)
    feature_max = X_train.quantile(.99)
    categorical_features = [x for x in X_test_sample.columns if x[:3] in CLEAR_settings.category_prefix]
    try:
        os.remove(CLEAR_settings.CLEAR_path + 'numericTemp.csv')
        os.remove(CLEAR_settings.CLEAR_path + 'categoricalTemp.csv')
    except OSError:
        pass
#create feature data for sensitivity files

    X_test_sample.reset_index(inplace=True, drop=True)

    for i in range(CLEAR_settings.first_obs, CLEAR_settings.last_obs + 1):

        for j in CLEAR_settings.numeric_features:
            sensitivity_num = 250
            te = np.tile(X_test_sample.iloc[i, :].values, [sensitivity_num, 1])
            te_c = X_test_sample.columns.get_loc(j)
            te[:, te_c] = np.linspace(feature_min[j], feature_max[j], sensitivity_num)
            f = open(CLEAR_settings.CLEAR_path + 'numericTemp.csv', 'a')
            np.savetxt(f, te, delimiter=',')
            f.close()

    if len(CLEAR_settings.category_prefix)!=0:
        for i in range(CLEAR_settings.first_obs, CLEAR_settings.last_obs + 1):
            for j in CLEAR_settings.category_prefix:
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
    if len(CLEAR_settings.class_labels)>2:
        if CLEAR_settings.multi_class_focus == 'All':
            num_class = len(CLEAR_settings.class_labels)
            multi_index = 0
        else:
            num_class = 1
            multi_index =[k for k, v in CLEAR_settings.class_labels.items() if v == CLEAR_settings.multi_class_focus][0]
        for c in range(num_class):
            predictions = model.predict_proba(sensit_df.values)
            temp_df = pd.DataFrame(columns=['observation', 'feature', 'newnn_class', 'probability', 'new_value'])
            cnt = 0
            for i in range(CLEAR_settings.first_obs, CLEAR_settings.last_obs + 1):
                for j in CLEAR_settings.numeric_features:
                    for k in range(sensitivity_num):
                        temp_df.loc[cnt, 'observation'] = i
                        temp_df.loc[cnt, 'feature'] = feature_list[(cnt // sensitivity_num) % len(CLEAR_settings.numeric_features)]
                        temp_df.loc[cnt, 'newnn_class'] = np.argmax(predictions[cnt])
                        temp_df.loc[cnt, 'probability'] = predictions[cnt,multi_index]
                        temp_df.loc[cnt, 'new_value'] = sensit_df.loc[cnt, j]
                        cnt += 1
            sensitivity_file = 'numSensitivity_m'+ str(multi_index) + '.csv'
            filename1 = CLEAR_settings.CLEAR_path + sensitivity_file
            temp_df.to_csv(filename1,index=False)
            multi_index +=1
    else:
        predictions = model.predict(sensit_df).flatten()
        sensitivity_file = 'numSensitivity' + '.csv'
        init_cnt = sensitivity_num *CLEAR_settings.first_obs*len(CLEAR_settings.numeric_features)
        cnt = 0
        top_row = ['observation', 'feature', 'probability', 'new_value']
        temp = len(CLEAR_settings.numeric_features)

        with open(CLEAR_settings.CLEAR_path + sensitivity_file, 'w',newline='') as file1:
            writes = csv.writer(file1, delimiter=',', skipinitialspace=True)
            writes.writerow(top_row)
            try:
                while cnt < len(predictions):
                    feature = CLEAR_settings.numeric_features[((cnt // sensitivity_num)) % temp]
                    observation = (cnt + init_cnt) // (sensitivity_num * temp)
                    #                        observation = cnt // (sensitivity_num * len(CLEAR_settings.numeric_features)) + 1
                    out_list = [observation, feature, predictions[cnt],
                                sensit_df.loc[cnt, feature]]
                    cnt += 1
                    writes.writerow(out_list)
            except:
                temp9 = 1
        file1.close()
    # then categorical features
        if len(CLEAR_settings.category_prefix)!=0:
            catSensit_df = pd.read_csv(CLEAR_settings.CLEAR_path + 'categoricalTemp.csv', header=None, names=feature_list)
            predictions = model.predict(catSensit_df).flatten()
            sensitivity_file = 'catSensitivity' + '.csv'
            init_cnt = CLEAR_settings.first_obs * len(categorical_features)
            cnt = 0
            top_row = ['observation', 'feature', 'probability', 'new_value']
            temp = len(categorical_features)

            with open(CLEAR_settings.CLEAR_path + sensitivity_file, 'w', newline='') as file1:
                writes = csv.writer(file1, delimiter=',', skipinitialspace=True)
                writes.writerow(top_row)
                try:
                    while cnt < len(predictions):
                        feature = categorical_features[(cnt % temp)]
                        observation = (cnt + init_cnt) //  temp
                        #                        observation = cnt // (sensitivity_num * len(CLEAR_settings.numeric_features)) + 1
                        out_list = [observation, feature, predictions[cnt],
                                    catSensit_df.loc[cnt, feature]]
                        cnt += 1
                        writes.writerow(out_list)
                except:
                    temp9 = 1
            file1.close()
    return

