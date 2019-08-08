from __future__ import print_function
import numpy as np
import pandas as pd
import tensorflow as tf
import csv
import os
import CLEAR_Process_Dataset,CLEAR_settings

def Create_sensitivity():
    np.random.seed(1)
    if (CLEAR_settings.case_study== 'PIMA Indians Diabetes') and (CLEAR_settings.test_sample==1):
        sensitivity_file ='PIMA_sensitivity.csv'     
    elif (CLEAR_settings.case_study== 'PIMA Indians Diabetes') and (CLEAR_settings.test_sample==2):
        sensitivity_file ='Final_diabetes_sensitivity2.csv'
    elif (CLEAR_settings.case_study== 'Credit Card') and (CLEAR_settings.test_sample==1):
        sensitivity_file ='Credit_card_sensitivity.csv'
    elif (CLEAR_settings.case_study== 'Credit Card') and (CLEAR_settings.test_sample==2):
        sensitivity_file ='Credit_card_sensitivity2.csv'
    elif (CLEAR_settings.case_study== 'Census') and (CLEAR_settings.test_sample==1):
        sensitivity_file ='Final_cens_sensitivity1.csv'
    elif (CLEAR_settings.case_study== 'Census') and (CLEAR_settings.test_sample==2):
        sensitivity_file ='Final_cens_sensitivity2.csv'
    elif (CLEAR_settings.case_study== 'BreastC') and (CLEAR_settings.test_sample==1):
        sensitivity_file ='BreastC_sensitivity1.csv'
    elif (CLEAR_settings.case_study== 'BreastC') and (CLEAR_settings.test_sample==2):
        sensitivity_file ='BreastC_sensitivity2.csv'    
    else:
        print('Case study / Test sample incorrectly specified')
        exit()  
    
    
    if CLEAR_settings.case_study == 'Credit Card':
        (X_train,X_test_sample,model, numeric_features, category_prefix,\
        feature_list)=CLEAR_Process_Dataset.Create_Credit_Datasets()
    elif CLEAR_settings.case_study == 'PIMA Indians Diabetes':
        (X_train,X_test_sample,model, numeric_features, category_prefix,\
        feature_list)=CLEAR_Process_Dataset.Create_PIMA_Datasets()
    elif CLEAR_settings.case_study == 'Census':
        (X_train, X_test_sample,model,numeric_features,category_prefix,\
        feature_list)=CLEAR_Process_Dataset.Create_Census_Datasets('sensitivity_program')
        numeric_features = ['age']
    elif CLEAR_settings.case_study == 'BreastC':
        (X_train, X_test_sample,model,numeric_features,category_prefix,\
        feature_list)=CLEAR_Process_Dataset.Create_BreastC_Datasets()   
    
    print('\n Performing grid search - step 1 of CLEAR method \n')
    s3_df=pd.DataFrame(columns=feature_list)
    feature_min= X_train.quantile(.01)
    feature_max= X_train.quantile(.99)
    
    try:
        os.remove(CLEAR_settings.CLEAR_path +'temp.csv')
    except OSError:
        pass
    
    X_test_sample.reset_index(inplace=True, drop=True)
    if CLEAR_settings.case_study !='Census':
        for i in range(CLEAR_settings.first_obs,CLEAR_settings.last_obs+1):
    
            for j in numeric_features:
                sensitivity_num=250
                te= np.tile(X_test_sample.iloc[i,:].values,[sensitivity_num,1])
                te_c=X_test_sample.columns.get_loc(j)
                te[:,te_c]=np.linspace(feature_min[j],feature_max[j],sensitivity_num)
                f= open(CLEAR_settings.CLEAR_path +'temp.csv', 'a')
                np.savetxt(f,te,delimiter=',')
                f.close() 
    else:
        for i, row in X_test_sample.iterrows():
            for j in numeric_features:
                sensitivity_num=250
                te= np.tile(X_test_sample.iloc[i,:].values,[sensitivity_num,1])
                te_c=X_test_sample.columns.get_loc(j)
                te[:,te_c]=np.linspace(feature_min[j],feature_max[j],sensitivity_num)
                f= open(CLEAR_settings.CLEAR_path +'temp.csv', 'a')
                np.savetxt(f,te,delimiter=',')
                f.close() 

  
    
    s3_df=pd.read_csv(CLEAR_settings.CLEAR_path +'temp.csv', header=None, names =feature_list)
               
    CLEAR_pred_input_func = tf.estimator.inputs.pandas_input_fn(
    x=s3_df,
    batch_size=5000,
    num_epochs=1,
    shuffle=False)
     #need to add org forecast. one thought nn does badly where it lacks data
    predictions = model.predict(CLEAR_pred_input_func)
    if CLEAR_settings.case_study !='Census':
        init_cnt=sensitivity_num*CLEAR_settings.first_obs*len(numeric_features)
    else:
        init_cnt =0
    cnt = 0
    top_row=['observation','feature','newnn_class','probability','new_value']

    with open(CLEAR_settings.CLEAR_path +sensitivity_file, 'w') as file1:
                writes = csv.writer(file1, delimiter=',',skipinitialspace=True)
                writes.writerow(top_row)
                try:
                    for p in predictions:
                        i= (cnt + init_cnt)//sensitivity_num
                        if i == 8:
                            temp9=1
                        j=feature_list[i%len(numeric_features)]
                        out_list= [i//len(numeric_features),j,p['class_ids'],p['probabilities'][1],s3_df.loc[cnt,j]]
                        cnt+=1
                        writes.writerow(out_list)
                except:
                    temp9=1
    file1.close()
   
# only a small proportion of observations have feasible w-perturbations
    if CLEAR_settings.case_study == 'Census':
        if (CLEAR_settings.case_study== 'Census') and (CLEAR_settings.test_sample==1):
            sensitivity_df = pd.read_csv(CLEAR_settings.CLEAR_path +'Final_cens_sensitivity1.csv')   
        else:
            sensitivity_df = pd.read_csv(CLEAR_settings.CLEAR_path +'Final_cens_sensitivity2.csv')
        temp_df=sensitivity_df.groupby(['observation','feature'])
        temp_df=temp_df['probability'].agg(['min','max'])
        sensitivity_idx=np.where((temp_df['min']<=0.5) & (temp_df['max']>0.5),1,0)    
        X_test_sample=X_test_sample[sensitivity_idx==1]
        X_test_sample.reset_index(inplace=True, drop=True)  

    return(X_train, X_test_sample,model,numeric_features,category_prefix,\
        feature_list)
