
import numpy as np
import CLEAR_settings
from sympy import symbols, solve, simplify
import pandas as pd
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt 
from scipy.signal import argrelextrema
import csv
from jinja2 import Environment, FileSystemLoader
from math import log10, floor


def Get_Counterfactual(sensitivity_df,feature, old_value, observation):
    temp_df= sensitivity_df.loc[(sensitivity_df['observation'] == observation) & (sensitivity_df['feature']==feature)]
    #& (t_df['new_value']==t_df['new_value'].min())]
    temp_df.reset_index(inplace=True, drop=True)    
    #get index of data point  i.e. where new_value - old_value = 0 (approx) 
    m=temp_df.iloc[(temp_df['new_value']-old_value).abs().argsort()[:1]].index[0]
    #get indexes of points minimum    
    n=abs(temp_df['probability']-0.5).values
    minimums=list(argrelextrema(n, np.less)[0])
    min_found= False
    while (len(minimums) !=0) and (min_found== False):
        nearest_min= minimums[abs(minimums-m).argmin()]
#check if min corresponds to 0.5:
        if temp_df.loc[nearest_min,'probability']>=0.5:
            if (temp_df.loc[nearest_min + 1,'probability']<0.5) \
                or (temp_df.loc[nearest_min - 1,'probability']<0.5):
                    min_found = True
            else:
                minimums.remove(nearest_min)      
        elif temp_df.loc[nearest_min,'probability']<0.5:
            if (temp_df.loc[nearest_min + 1,'probability']>0.5) \
                or (temp_df.loc[nearest_min - 1,'probability']>0.5):
                    min_found = True
            else:
                minimums.remove(nearest_min)      
        else:
             print('error in probability data')

    if min_found == False:
         perc50 = np.NaN
    else:    

        a= temp_df.loc[nearest_min,'probability']
        b= temp_df.loc[nearest_min,'new_value']
        if temp_df.loc[nearest_min,'probability']>=0.5:
            if temp_df.loc[nearest_min + 1,'probability']<0.5:           
                c= temp_df.loc[nearest_min + 1,'probability']
                d= temp_df.loc[nearest_min + 1,'new_value']
            else:
                c= temp_df.loc[nearest_min - 1,'probability']
                d= temp_df.loc[nearest_min - 1,'new_value']
            perc50=(b-d)* ((0.5-c)/(a-c)) + d
# a= nearest higer probability to 0.5, and b = corresponding 'new_value' for the feature
# c= nearest lower probability to 0.5 and d = corresponding 'new_value' for the feature
# then 50th percentile = (b-d)*((0.5-c)/(a+c))+d
        elif temp_df.loc[nearest_min,'probability']<0.5:
            if temp_df.loc[nearest_min + 1,'probability']>0.5:         
                c= temp_df.loc[nearest_min + 1,'probability']
                d= temp_df.loc[nearest_min + 1,'new_value']
            else:
                c= temp_df.loc[nearest_min - 1,'probability']
                d= temp_df.loc[nearest_min - 1,'new_value']
        perc50=(b-d)* ((0.5-c)/(a-c)) + d
    return (perc50)




def Calculate_Perturbations(explainer, results_df,sensitivity_df,\
                            feature_list,numeric_features, model):    
      
    """ Counterfactual perturbations are now calculated and stored
        in the nncomp_df dataframe. If CLEAR calculates a perturbation
        that is infeasible, then the details of the perturbation
        are stoted in the missing_log_df dataframe. CLEAR will classify
        a perturbation as bdeing infeasible if it is outside 'the feasibility
        range' it calculates for each variable.
    """    
    print("\n Calculating w-counterfactuals \n")
    nncomp_df = pd.DataFrame(columns=['observation','feature','weight','reg_prob','oldnn_prob','newnn_class',
                                      'prob_with_new_value','old_value','new_value'])
    
    
    missing_log_df = pd.DataFrame(columns=['observation','feature','reason','perturbation'])
    if CLEAR_settings.with_indicator_feature == True:
    #temp added Glucose_old_sqrd. delete this in poly_df
        feature_list.append('IndicatorFeature')
        feature_list.append('IndicatorFeature_sqrd')
        for i in range (CLEAR_settings.first_obs,CLEAR_settings.last_obs+1):
            CLEAR_settings.indicatorFeature = np.where(( results_df.local_data[i][1]>=CLEAR_settings.indicator_threshold), 1, 0)
            temp=np.append(results_df.local_data[i], CLEAR_settings.indicatorFeature)
            #below is for Glucose_oild_sqrd to be deleted
            temp=np.append(temp, CLEAR_settings.indicatorFeature)
            results_df.local_data[i]= temp        
            
    for i in range (CLEAR_settings.first_obs,CLEAR_settings.last_obs+1):
        s1=pd.Series(results_df.local_data[i],feature_list)
        s2=pd.DataFrame(columns=feature_list)
        s2=s2.append(s1,ignore_index=True)
        x= symbols('x')
        str_eqn = ""
        raw_eqn= results_df.loc[i,'features'].copy()
        raw_weights = results_df.loc[i,'weights'].tolist()
        raw_data = results_df.loc[i,'local_data'].tolist()
        features_processed = 0
#the next 2 lines ensures that the same code can be used irrespective of whether the regression has
#been forced through the data point to be explained
        if raw_eqn[0]=='1':
            raw_eqn.remove('1')
            raw_weights.pop(0)
        for j in range(0,len(feature_list)):
              features_processed +=1 
              if features_processed >1 and CLEAR_settings.perturb_one_feature == True:
                  break
              old_value = s2.iloc[0,j]
              target_feature_p_value =""
              target_feature_se_value =""
              if (CLEAR_settings.case_study == 'Census') and (CLEAR_settings.perturb_one_feature == True):    
                 target_feature= CLEAR_settings.only_feature_perturbed
                 if not any(target_feature in s for s in raw_eqn):
                     if missing_log_df.empty:
                        idx = 0
                     else:
                        idx=missing_log_df.index.max() + 1
                        missing_log_df.loc[idx,'observation']= i
                        missing_log_df.loc[idx,'reason']='not in formula'
                     break
              else:
                  target_feature = feature_list[j]
                  if not any(target_feature in s for s in numeric_features):
                      continue
                  temp_df=sensitivity_df[(sensitivity_df['observation']==i) & (sensitivity_df['feature']==target_feature)]
                  temp_df=temp_df['probability'].agg(['min','max'])
                  if not (temp_df['min']<=0.5) & (temp_df['max']>0.5):    
                      continue    
              #establish if feature is in equation
              if not any(target_feature in s for s in raw_eqn):
                    if missing_log_df.empty:
                        idx = 0
                    else:
                        idx=missing_log_df.index.max() + 1
                    missing_log_df.loc[idx,'observation']= i
                    missing_log_df.loc[idx,'feature']= target_feature
                    missing_log_df.loc[idx,'reason']='not in raw equation'
                    continue
              else:  
                  
                if CLEAR_settings.LIME_comparison == True:
                    explainer.feature_min={}
                    explainer.feature_max={}
                    explainer.feature_min[target_feature] =-3
                    explainer.feature_max[target_feature] = 3
                  
                if CLEAR_settings.regression_type=='multiple':
                    str_eqn= '-0.5 +'+str(results_df.loc[i,'intercept'])
                else:    
                    str_eqn=str(results_df.loc[i,'intercept'])
                for raw_feature in raw_eqn:
                    if raw_feature == '1':
                        pass
                    elif raw_feature == 'IndicatorFeature':
                        str_eqn +=  "+" + str(raw_weights[raw_eqn.index(raw_feature)])+"*IndicatorFeature"
                    elif raw_feature == 'IndicatorFeature_sqrd':
                        str_eqn +=  "+" + str(raw_weights[raw_eqn.index(raw_feature)])+"*IndicatorFeature**2"
                    elif raw_feature ==    "IndicatorFeature_" + CLEAR_settings.feature_with_indicator:  #   'Glucose_IndicatorFeature':                            
                        new_term = str(raw_weights[raw_eqn.index("IndicatorFeature_" \
                                        + CLEAR_settings.feature_with_indicator)])+"*x*IndicatorFeature"
                        str_eqn += "+ " + new_term
                    elif raw_feature ==     CLEAR_settings.feature_with_indicator +"_IndicatorFeature":  #   'Glucose_IndicatorFeature':                            
                        new_term = str(raw_weights[raw_eqn.index(CLEAR_settings.feature_with_indicator + \
                                    "_IndicatorFeature")])+"*x*IndicatorFeature"
                        str_eqn += "+ " + new_term    
                    elif ('_' in raw_feature) &  ('IndicatorFeature' in raw_feature):
                        interaction_terms = raw_feature.split('_')
                        if interaction_terms[0]=='IndicatorFeature':
                           new_term = str(raw_data[feature_list.index(interaction_terms[1])] \
                                    *raw_weights[raw_eqn.index(raw_feature)])+"*IndicatorFeature"  
                        elif  interaction_terms[1]== 'IndicatorFeature':  
                            new_term = str(raw_data[feature_list.index(interaction_terms[0])] \
                                    *raw_weights[raw_eqn.index(raw_feature)])+"*IndicatorFeature"
                        str_eqn += "+ " + new_term                                
                      
                        
           
             
                    elif raw_feature == target_feature:
                            str_eqn +=  "+" + str(raw_weights[raw_eqn.index(raw_feature)])+"*x"
                            target_feature_weight = raw_weights[raw_eqn.index(raw_feature)]
                    elif raw_feature in feature_list:
                        new_term= raw_data[feature_list.index(raw_feature)]*raw_weights[raw_eqn.index(raw_feature)]
                        str_eqn += "+ " + str(new_term)
                    elif raw_feature == str(target_feature)+ "_sqrd":
                        str_eqn += "+" + str(raw_weights[raw_eqn.index(raw_feature)]) +"*x**2"
                    elif raw_feature.endswith('_sqrd'):
                        new_term = raw_feature.replace('_sqrd','')
                        new_term= (raw_data[feature_list.index(new_term)]**2)*raw_weights[raw_eqn.index(raw_feature)]
                        str_eqn += "+ " + str(new_term)
                    #missing x in interaction terms
                    elif '_' in raw_feature:
                        interaction_terms = raw_feature.split('_')
                        if interaction_terms[0]==target_feature:
                           new_term = str(raw_data[feature_list.index(interaction_terms[1])] \
                                    *raw_weights[raw_eqn.index(raw_feature)])+"*x"  
                        elif  interaction_terms[1]==target_feature:  
                            new_term = str(raw_data[feature_list.index(interaction_terms[0])] \
                                    *raw_weights[raw_eqn.index(raw_feature)])+"*x"  
                        else:                                                
                            new_term = str(raw_data[feature_list.index(interaction_terms[0])] \
                                    *raw_data[feature_list.index(interaction_terms[1])] \
                                    *raw_weights[raw_eqn.index(raw_feature)])
                        str_eqn += "+ " + new_term
                    else:
                        print("error in processing equation string")            
                try:
                    if 'x' not in str(simplify(str_eqn)):
                        if missing_log_df.empty:
                            idx = 0
                        else:
                            idx=missing_log_df.index.max() + 1
                        missing_log_df.loc[idx,'observation']= i
                        missing_log_df.loc[idx,'feature']= target_feature
                        missing_log_df.loc[idx,'reason']='not in simplified equation'
                        continue
                except:    
                        temp9=1
                if 'IndicatorFeature' in str(simplify(str_eqn)):              
                    if target_feature == CLEAR_settings.feature_with_indicator:                   
    #CLEAR calculates perturbations for the 2  IndicatorFeature_sqrd','1')
                       str_eqn_0= str_eqn.replace('IndicatorFeature_sqrd','1')    
                       str_eqn_1 = str_eqn_0.replace('IndicatorFeature','1')
                       eqn_roots = solve(str_eqn_1,x)
                       temp = []
                       valid_roots=[]
                       for k in eqn_roots:
                           if k.is_real:
                               if k >= CLEAR_settings.indicator_threshold:
                                   temp.append(k)
                           else:
                               if k == eqn_roots[len(eqn_roots)-1]:
                                  if missing_log_df.empty:
                                      idx = 0
                                  else:
                                    idx=missing_log_df.index.max() + 1
                                  missing_log_df.loc[idx,'feature']= target_feature
                                  missing_log_df.loc[idx,'observation']= i
                                  missing_log_df.loc[idx,'reason']='value not real' 
                                  new_value = None
                                  continue        
                        # get minimum perturbation                        
                       str_eqn_0 = str_eqn.replace('IndicatorFeature_sqrd','0')
                       str_eqn_1 = str_eqn_0.replace('IndicatorFeature','0')
                       eqn_roots = solve(str_eqn_1,x)
                       for k in eqn_roots:
                           if k.is_real:
                              if k< CLEAR_settings.indicator_threshold:
                                  temp.append(k)
                           elif k == eqn_roots[len(eqn_roots)-1]:
                              if missing_log_df.empty:
                                  idx = 0
                              else:
                                idx=missing_log_df.index.max() + 1
                              missing_log_df.loc[idx,'feature']= target_feature
                              missing_log_df.loc[idx,'observation']= i
                              missing_log_df.loc[idx,'reason']='value not real' 
                              new_value = None
                              continue
                        # get minimum perturbation
                       if len(temp)>0:
                       #check if any of the perturbations is feasible, if so discard
                       # infeasible perturbations
                            temp2 = []
                            for y in temp:
                               if explainer.feature_min[target_feature] <= y <=explainer.feature_max[target_feature]:
                                  temp2.append(y) 
                            if len(temp2)>0:
                               valid_roots = temp2 -old_value
                               new_value = min(valid_roots, key=abs)
                               new_value = new_value + old_value
                            else:
                          # if roots are all infeasible, take root nearest to feasibility range
                              j=0
                              lowest_root = 999
                              for y in temp:      
                                  k= min(abs(explainer.feature_min[target_feature] -y),abs(explainer.feature_max[target_feature] -y))  
                                  if k<lowest_root:
                                      lowest_root = k
                                      new_value = y                                
                       else:
                           continue
    #now consider the cases where IndicatorFeature and another feature
                    else:                   
                       
                       if raw_data[feature_list.index(CLEAR_settings.feature_with_indicator)] >= CLEAR_settings.indicator_threshold:
                            str_eqn_0 = str_eqn.replace('IndicatorFeature_sqrd','1')
                            str_eqn_1 = str_eqn_0.replace('IndicatorFeature','1')
                            eqn_roots = solve(str_eqn_1,x)
                       else:
                            str_eqn_0 = str_eqn.replace('IndicatorFeature_sqrd','0')
                            str_eqn_1 = str_eqn_0.replace('IndicatorFeature','0')
                            eqn_roots = solve(str_eqn_1,x)
                       temp = []
                       for k in eqn_roots:
                           if k.is_real:
                               temp.append(k)
                           elif k == eqn_roots[len(eqn_roots)-1]:
                              if missing_log_df.empty:
                                  idx = 0
                              else:
                                idx=missing_log_df.index.max() + 1
                              missing_log_df.loc[idx,'feature']= target_feature
                              missing_log_df.loc[idx,'observation']= i
                              missing_log_df.loc[idx,'reason']='value not real' 
                              new_value = None
                              continue
                        # get minimum perturbation
                       if len(temp)>0:
                            temp2 = []
                            for y in temp:
                               if explainer.feature_min[target_feature] <= y <=explainer.feature_max[target_feature]:
                                  temp2.append(y) 
                            if len(temp2)>0:
                               valid_roots = temp2 -old_value
                               new_value = min(valid_roots, key=abs)
                               new_value = new_value + old_value
                            else:
                          # if roots are all infeasible, take root nearest to feasibility range
                              lowest_root = 999
                              for y in temp:      
                                  k= min(abs(explainer.feature_min[target_feature] -y),abs(explainer.feature_max[target_feature] -y))  
                                  if k<lowest_root:
                                      lowest_root = k
                                      new_value = y                                
                       else:
                           continue




    #now consider cases without IndicatorFeature
                else:           
                        temp = []
                        eqn_roots = solve(str_eqn,x)
                        for k in eqn_roots:
                           if k.is_real:
                                  temp.append(k)
                               
                           elif k == eqn_roots[len(eqn_roots)-1]:
                              if missing_log_df.empty:
                                  idx = 0
                              else:
                                idx=missing_log_df.index.max() + 1
                              missing_log_df.loc[idx,'feature']= target_feature
                              missing_log_df.loc[idx,'observation']= i
                              missing_log_df.loc[idx,'reason']='value not real' 
                              new_value = None
                              continue
                        # get minimum perturbation
                        if len(temp)>0:
                            temp2 = []
                            for y in temp:
                               if explainer.feature_min[target_feature] <= y <=explainer.feature_max[target_feature]:
                                  temp2.append(y) 
                            if len(temp2)>0:
                               valid_roots = temp2 -old_value
                               new_value = min(valid_roots, key=abs)
                               new_value = new_value + old_value
                            else:
                          # if roots are all infeasible, take root nearest to feasibility range
                              lowest_root = 999
                              for y in temp:      
                                  k= min(abs(explainer.feature_min[target_feature] -y),abs(explainer.feature_max[target_feature] -y))  
                                  if k<lowest_root:
                                      lowest_root = k
                                      new_value = y
                              if lowest_root ==999:
                                  continue
                 #takes the minimum perturbation from both sides of Indicator threshold. If possible select
                 # feasible perturbation
                        else:
                            continue
    
                new_value=np.float64( new_value)
                s2.iloc[0,j]=new_value          
                nncomp_idx= i*10+j
    
    
                CLEAR_pred_input_func = tf.estimator.inputs.pandas_input_fn(
                      x=s2,
                      batch_size=1,
                      num_epochs=1,
                      shuffle=False)

                predictions = model.predict(CLEAR_pred_input_func)
                for p in predictions:
                    nncomp_df.loc[nncomp_idx,'observation'] = i
                    nncomp_df.loc[nncomp_idx,'feature'] = target_feature
                    nncomp_df.loc[nncomp_idx,'reg_prob'] = results_df.loc[i,'reg_prob']
                    nncomp_df.loc[nncomp_idx,'regression_class']=results_df.loc[i,'regression_class']
                    nncomp_df.loc[nncomp_idx,'prob_with_new_value']= p['probabilities'][1]
                    nncomp_df.loc[nncomp_idx,'oldnn_prob']= results_df.loc[i,'nn_forecast']
                    nncomp_df.loc[nncomp_idx,'newnn_class']=p['class_ids']
                    nncomp_df.loc[nncomp_idx,'old_value']= old_value
                    nncomp_df.loc[nncomp_idx,'new_value']= new_value
                    try:
                        nncomp_df.loc[nncomp_idx,'p_value']= target_feature_p_value
                        nncomp_df.loc[nncomp_idx,'weight']= target_feature_weight
                        nncomp_df.loc[nncomp_idx,'se']= target_feature_se
                    except:
                        pass
                s2.iloc[0,j]= old_value
                if CLEAR_settings.perturb_one_feature == True:
                    break
    nncomp_df.observation =nncomp_df.observation.astype(int)
    nncomp_df.reset_index(inplace=True, drop=True)         
    
    """
    Determines the actual values of the MPL decision boundary. This will then be used 
    for determining the fidelity errors of the CLEAR perturbations.
    """
    
    temp_df=sensitivity_df.groupby(['observation','feature'])
    temp_df=temp_df['probability'].agg(['min','max'])
    sensitivity_idx=np.where((temp_df['min']<=0.5) & (temp_df['max']>0.5),1,0)    
    percentiles = []
    for h in nncomp_df.index:
        i =nncomp_df.loc[h,'observation']
        if CLEAR_settings.case_study == 'Census':
            i= np.where(sensitivity_idx == 1)[0][i]
       # change to feature = nncomp_df.loc[h,'feature']????
        g = nncomp_df.loc[h,'feature']
        old_value = nncomp_df.loc[h,'old_value']
      
    #estimate new_value corresponding to 50th percentile. This Gridsearch assumes only a single 50th percentile
        perc50= Get_Counterfactual(sensitivity_df,g, old_value, i)    
        percentiles.append(perc50)
    
    """ Create summary statistics and frquency histogram
    """
    nncomp_df['perc50']=percentiles
    nncomp_df['accuracy']=np.where(pd.isna(nncomp_df['perc50']),np.NaN,abs(nncomp_df['new_value']-nncomp_df['perc50']))
    
    less_target_sd=0
    if nncomp_df['accuracy'].count()!=0:
        less_target_sd=nncomp_df[nncomp_df.accuracy<=0.25].accuracy.count()
        x= nncomp_df['accuracy']
        x = x[~x.isna()]
        ax= x.plot.hist(grid=True, bins=20, rwidth=0.9)
        if CLEAR_settings.LIME_comparison== False:
            plt.title('perturbations = '+str(nncomp_df['accuracy'].count())+'  Freq Counts <= 0.25 sd = ' + str(less_target_sd)
                         + '\n' + 'regression = ' + CLEAR_settings.regression_type +  ', score = ' + CLEAR_settings.score_type 
                         + ', sample = ' + str(CLEAR_settings.num_samples)
                         +'\n' + 'max_predictors = ' + str(CLEAR_settings.max_predictors)
                         + ', regression_sample_size = ' +  str(CLEAR_settings.regression_sample_size))
        else:
            plt.title('perturbations = '+str(nncomp_df['accuracy'].count())+'  Freq Counts <= 0.25 sd = ' + str(less_target_sd))
        plt.xlabel('Standard Deviations')
        fig = ax.get_figure()
        fig.savefig(CLEAR_settings.CLEAR_path +'hist'+  datetime.now().strftime("%Y%m%d-%H%M")+'.png',bbox_inches = "tight")   
    else:
        print('no data for plot')
    #x=np.array(nncomp_df['accuracy'])
    
                    
    filename1 = CLEAR_settings.CLEAR_path +'wPerturb_'+ datetime.now().strftime("%Y%m%d-%H%M")+'.csv'   
    nncomp_df.to_csv(filename1)    
    filename2 = CLEAR_settings.CLEAR_path +'missing_'+ datetime.now().strftime("%Y%m%d-%H%M")+'.csv'
    missing_log_df.to_csv(filename2)  
    output= [CLEAR_settings.case_study, less_target_sd]
    if CLEAR_settings.LIME_comparison== True:
        filename3 = 'LIMEbatch.csv'
    else:
        filename3 = 'batch.csv'
    try:
        with open(CLEAR_settings.CLEAR_path + filename3 , 'a') as file1:
            writes = csv.writer(file1, delimiter=',',skipinitialspace=True)    
            writes.writerow(output)
        file1.close()
    except:
        pass
    return(nncomp_df)



def Single_prediction_report(results_df,nncomp_df,regression_obj,feature_list):
#dataframe to HTML Report
    if CLEAR_settings.case_study=='Census':
        explanandum = 'earning > $50k'
    elif CLEAR_settings.case_study=='PIMA Indians Diabetes':
        explanandum = 'diabetes'        
    elif CLEAR_settings.case_study=='Credit Card':
        explanandum = 'default'        
    else:
        explanandum = 'breast cancer'          
    
    def round_sig(x, sig=2):
        return round(x, sig-int(floor(log10(abs(x))))-1)
    j= results_df.index.values[0]
    if CLEAR_settings.regression_type== 'multiple':
        regression_formula = 'prediction = ' +  str(round_sig(results_df.intercept[j]))
    else:
        regression_formula = '<font size = "4.5">prediction =  [ 1 + e<sup><b>-w<sup>T</sup>x</sup></b> ]<sup> -1</sup></font size><br><br>' \
         + '<font size = "4.5"><b><i>w</i></b><sup>T</sup><b><i>x</font size></i></b> =  ' +  str(round_sig(results_df.intercept[j]))

    for i in range(len(results_df.features[j])):    
        if results_df.features[j][i] == '1':
            continue
        elif results_df.weights[j][i]<0:
                regression_formula = regression_formula + ' - '+ str(-1*round_sig(results_df.weights[j][i])) + \
                                     ' ' + results_df.features[j][i]        
        else:
                regression_formula = regression_formula + ' + ' + str(round_sig(results_df.weights[j][i])) + \
                                     ' ' + results_df.features[j][i]                                     
    regression_formula= regression_formula.replace("_sqrd"," sqrd")
    regression_formula= regression_formula.replace("_","*")
    report_AI_prediction = str(round_sig(results_df.nn_forecast[j]))
    if CLEAR_settings.score_type == 'adjR':  
        regression_score_type = "Adjusted R-Squared"
    else:
        regression_score_type = CLEAR_settings.score_type
        
        
    HTML_df= pd.DataFrame(columns=['feature','input value','coeff','abs_coeff'])
    report_selected = [w.replace('_sqrd', ' sqrd') for w in results_df.features[j]]
    report_selected = [w.replace('_', '*') for w in report_selected]
#   results_df.spreadsheet_data does not have intercept data
    sp_correction = 0
    for i in range(len(results_df.features[j])):            
        feature =results_df.features[j][i] 
        if feature == '1':
            sp_correction = 1
            continue
        else:
            HTML_df.loc[i,'feature']= results_df.features[j][i] 
            HTML_df.loc[i,'input value']= results_df.spreadsheet_data[j][i-sp_correction]
            HTML_df.loc[i,'coeff']=results_df.weights[j][i]
            HTML_df.loc[i,'abs_coeff']=abs(results_df.weights[j][i])

        
    HTML_df=HTML_df.sort_values(by=['abs_coeff'], ascending = False)
    HTML_df.drop(['abs_coeff'],axis =1,inplace=True)
    
    HTML_df=HTML_df.head(10)
        
    counter_df =nncomp_df[['feature','old_value','perc50']].copy()
    counter_df.rename(columns={'old_value':'input value', 'perc50':'counterfactual value'}, inplace=True )
#    HTML_df.to_html('CLEAR.HTML')
    
    nncomp_df['error']= nncomp_df['new_value']-nncomp_df['perc50']
    reg_counter_df=nncomp_df[['feature','new_value','error']].copy()
    reg_counter_df.error= abs(reg_counter_df.error)
    reg_counter_df.rename(columns={'new_value':'counterfactual value',\
                                   'error':'fidelity error'}, inplace=True )


    # results_df.weights needs pre-processing prior to sending to HTML
    weights=results_df.weights.values[0]
    weights=weights.tolist()
    pd.set_option('colheader_justify', 'left','precision', 2)
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template("newCLEAR_report.html")
    template_vars = {"title" : "CLEAR Statistics",
                     "regression_table": HTML_df.to_html(index=False, classes = 'mystyle'),
                     "counterfactual_table": counter_df.to_html(index=False,classes='mystyle'),
#                     "inaccurate_table": inaccurate_df.to_html(index=False,classes='mystyle'),
                     "dataset_name": CLEAR_settings.case_study,
                     "explanadum":explanandum,
                     "observation_number": j,
                     "regression_formula": regression_formula,
                     "prediction_score": round_sig(results_df.Reg_Score[j]),
                     "regression_score_type": regression_score_type,
                     "regression_type":CLEAR_settings.regression_type,
                     "AI_prediction":report_AI_prediction,
                     "reg_counterfactuals":reg_counter_df.to_html(index=False, classes = 'mystyle'),
                     "feature_list": feature_list,
                     "spreadsheet_data":results_df.spreadsheet_data.values[0],
                     "weights":weights,
                     "intercept":results_df.intercept.values[0]
                     }
    # Render our file and create the PDF using our css style file
    #html_out = template.render(template_vars)
    with open('CLEAR_prediction_report.html', 'w') as fh:
        fh.write(template.render(template_vars))

    fig = plt.figure()
    plt.scatter(regression_obj.neighbour_df.loc[:,'prediction'] ,regression_obj.untransformed_predictions , c='green',s=10)
    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), c= "red", linestyle='-')
    
    plt.xlabel('Target AI System')
    if CLEAR_settings.regression_type == 'logistic':
        plt.ylabel('CLEAR Logistics Regression')
    elif  CLEAR_settings.regression_type == 'multiple':   
         plt.ylabel('CLEAR Multiple Regression')
    else:
         plt.ylabel('CLEAR Polynomial Regression')
    
    fig.savefig('CLEAR_plot.png', bbox_inches = "tight")
    
    
    return()