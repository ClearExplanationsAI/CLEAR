from datetime import datetime
import pandas as pd
import numpy as np
import CLEAR_settings, CLEAR_Process_Dataset, CLEAR_perturbations
import re

#This runs CLEAR using the neighbourhood dataset generation and regression algorithms of LIME
def LIME_CLEAR(X_test_sample,explainer,sensitivity_df,feature_list,numeric_features, model):

    results_df = pd.DataFrame(columns=['Reg_score','intercept', 'features','weights',\
                                       'nn_forecast','reg_prob','regression_class',\
                                       'spreadsheet_data','local_data','accuracy'])
    observation_num = CLEAR_settings.first_obs       
    feature_list = X_test_sample.columns.tolist()
    for i in range(CLEAR_settings.first_obs,CLEAR_settings.last_obs+1):       
        data_row=pd.DataFrame(columns=feature_list)
        data_row=data_row.append(X_test_sample.iloc[i],ignore_index=True)
        data_row.fillna(0, inplace= True)
        
        if CLEAR_settings.case_study ==  'Credit Card':
            LIME_datarow, LIME_features = CLEAR_Process_Dataset.Credit_categorical(data_row)    
            LIME_datarow= LIME_datarow.flatten()        
            lime_out= explainer.explain_instance(LIME_datarow, model, num_features=CLEAR_settings.max_predictors,\
                                                 num_samples=CLEAR_settings.LIME_sample)
        elif CLEAR_settings.case_study ==  'Census':   
            LIME_datarow, LIME_features = CLEAR_Process_Dataset.Adult_categorical(data_row)    
            LIME_datarow= LIME_datarow.flatten()        
            lime_out= explainer.explain_instance(LIME_datarow, model, num_features=CLEAR_settings.max_predictors, \
                                                 num_samples=CLEAR_settings.LIME_sample)

        else:
            lime_out= explainer.explain_instance(X_test_sample.iloc[i].values, model, num_features=CLEAR_settings.max_predictors,\
                                                 num_samples=CLEAR_settings.LIME_sample)
        coeffs = np.asarray([x[1] for x in lime_out.local_exp[1]])        
        feature_idx=np.asarray([x[0] for x in lime_out.local_exp[1]]) 
        features =[lime_out.domain_mapper.exp_feature_names[x] for x in feature_idx]
        if CLEAR_settings.case_study ==  'Credit Card':
            str1 = ','.join(features)
            str1= str1.replace("MARRIAGE=","marDd")
            str1= str1.replace("EDUCATION=","eduDd")
            str1= str1.replace("SEX=","genDd")
            features = str1.split(",")
        if CLEAR_settings.case_study ==  'Census':      
            str1 = ','.join(features)       
            rep={'EDUCATION=0':'eduDdBachelors', 'EDUCATION=1':'eduDdCommunityCollege', \
                 'EDUCATION=2':'eduDdDoctorate','EDUCATION=3':'eduDdHighGrad', \
                 'EDUCATION=4':'eduDdMasters','EDUCATION=5':'eduDdProfSchool', \
                 'EDUCATION=6':'eduDddropout','Occupation=0':'occDdBlueCollar',\
                 'Occupation=1':'occDdExecManagerial','Occupation=2':'occDdProfSpecialty',\
                 'Occupation=3':'occDdSales', 'Occupation=4':'occDdServices',\
                 'Work=1':'workDdGov', 'Work=2':'workDdPrivate',\
                 'MARRIAGE=1':'marDdmarried', 'MARRIAGE=2':'marDdnotmarried',\
                 'SEX=1':'genDdFemale', 'SEX=2':'genDdMale'}        
            rep = dict((re.escape(k), v) for k, v in rep.items())
            pattern = re.compile("|".join(rep.keys()))
            str1 = pattern.sub(lambda m: rep[re.escape(m.group(0))], str1)
            features = str1.split(",")
            
            
        print('Processed observation ' + str(i))
        results_df.at[i,'features'] = features
        results_df.loc[i,'Reg_score'] = lime_out.score
        results_df.loc[i,'nn_forecast'] = lime_out.predict_proba[1]     
        results_df.loc[i,'reg_prob'] = lime_out.local_pred[0]  
        results_df.loc[i,'regression_class'] = 'x'
        results_df.at[i,'spreadsheet_data'] ='x'
        results_df.at[i,'local_data'] =data_row.values[0]
        results_df.loc[i,'accuracy'] = 'x'
        results_df.loc[i,'intercept'] = lime_out.intercept[1]
        results_df.at[i,'weights'] = coeffs
    
        observation_num += 1  
    filename1 = CLEAR_settings.CLEAR_path +'LIME_'+ datetime.now().strftime("%Y%m%d-%H%M")+'.csv'   
    results_df.to_csv(filename1)
    #    results_df.to_pickle(filename2) 
     
      
    """ Counterfactual perturbations are now calculated and stored
        in the nncomp_df dataframe. If CLEAR calculates a perturbation
        that is infeasible, then the details of the perturbation
        are stoted in the missing_log_df dataframe. CLEAR will classify
        a perturbation as being infeasible if it is outside 'the feasibility
        range' it calculates for each variable.
    """    
    CLEAR_perturbations.Calculate_Perturbations(explainer, results_df,sensitivity_df,feature_list,numeric_features, model)
    return()