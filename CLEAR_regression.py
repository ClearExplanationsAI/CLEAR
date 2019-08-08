"""
Functions for CLEAR to create local regressions
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn
import sklearn.preprocessing
import statsmodels.formula.api as sm
import sys
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures
import CLEAR_settings,CLEAR_perturbations
""" specify input parameters"""
 

kernel_type = 'Euclidean' # sets distance measure for the neighbourhood algorithms


""" Creates Synthetic Dataset and sets some summary statistics for the variables
    in the MPL training dataset """
    
class CLEARExplainer(object):
#Gets min and max needed for generating synthetic data
# 'dataset_for_regressionsset' is equal to X_train for PIMA Indian Datasert
#  X_neighborhood for Census/Adult
    def __init__(self, X_train, model,numeric_features,\
                  category_prefix,feature_list,sensitivity_df):
        self.feature_min= X_train.quantile(.01)
        self.feature_max= X_train.quantile(.99)
        self.model=model
        self.feature_list = feature_list
        self.num_features =len(feature_list)
        self.sensitivity_df = sensitivity_df

#creates synthetic data
        if CLEAR_settings.case_study in ('PIMA Indians Diabetes','BreastC'):           
            self.master_df = pd.DataFrame(np.zeros(shape=(CLEAR_settings.num_samples,\
                                        self.num_features)), columns=numeric_features)
            for i in numeric_features:
                self.master_df.loc[:,i] = np.random.uniform(self.feature_min[i], \
                                   self.feature_max[i], CLEAR_settings.num_samples)  
# the synthetic data is now 'labelled' by the MLP model    
            CLEAR_pred_input_func = tf.estimator.inputs.pandas_input_fn(
                  x=self.master_df,
                  batch_size=1,
                  num_epochs=1,
                  shuffle=False)
            predictions = model.predict(CLEAR_pred_input_func)
    
            y=np.array([])
            for p in predictions:
               y= np.append(y, p['probabilities'][1])
            self.master_df['prediction'] = y
        
        elif CLEAR_settings.case_study in ('Census','Credit Card'):
            self.master_df = pd.DataFrame(np.zeros(shape=(CLEAR_settings.num_samples, \
                                self.num_features)), columns=X_train.columns.tolist())
            for prefix in category_prefix:
                cat_cols = [col for col in X_train if col.startswith(prefix)]    
                t=X_train[cat_cols].sum()
                st=t.sum()
                ut=t.cumsum()
                pt= t/st
                ct= ut/st
                if len(cat_cols) >1:
                    cnt=0
                    for cat in cat_cols:
                        if cnt == 0:
                            self.master_df[cat]=np.random.uniform(0, 1,CLEAR_settings.num_samples)  
                            self.master_df[cat]=np.where(self.master_df[cat]<=pt[cat],1,0)
                        elif cnt== len(cat_cols)-1:
                            self.master_df[cat]=self.master_df[cat_cols].sum(axis=1)
                            self.master_df[cat]= np.where(self.master_df[cat]==0,1,0)
                        else:
                            self.master_df.loc[self.master_df[cat_cols].sum(axis=1)==1,cat]=99
                            v=CLEAR_settings.num_samples-self.master_df[self.master_df[cat_cols].sum(axis=1)>99].shape[0]
                            self.master_df.loc[self.master_df[cat_cols].sum(axis=1)==0,cat] \
                                    =np.random.uniform(0, 1,v)  
                            self.master_df[cat]=np.where(self.master_df[cat]<=(pt[cat]/(1-ct[cat]+pt[cat])),1,0)
                        cnt+=1
            for i in numeric_features:
                self.master_df.loc[:,i] = np.random.uniform(self.feature_min[i], \
                                   self.feature_max[i], CLEAR_settings.num_samples)  
# the synthetic data is now 'labelled' by the MLP model    
            CLEAR_pred_input_func = tf.estimator.inputs.pandas_input_fn(
                  x=self.master_df,
                  batch_size=5000,
                  num_epochs=1,
                  shuffle=False)
            predictions = model.predict(CLEAR_pred_input_func)
        
            y=np.array([])
            for p in predictions:
               y= np.append(y, p['probabilities'][1])
            self.master_df['prediction'] = y     
        
        else:
            sys.exit("case study incorrectly specified")

#        if CLEAR_settings.neighbourhood_algorithm != 'experiment':
#            save_master_df=True 
#            if save_master_df==True:
#                self.master_df.to_csv(CLEAR_settings.CLEAR_path+'master_df.csv')

        
    def explain_data_point(self, data_row, observation_num):    
        #data row needs to be a dataframe or dictionary not series
        self.observation_num = observation_num
        self.data_row = data_row
        #set to 6 for no centering when using logistic regression
        self.additional_weighting = 3
        #temp fix to prevent long Credit Card runs
        if CLEAR_settings.no_centering == True:
            self.additional_weighting = 6        
        self.local_df = self.master_df.copy(deep=True)
        CLEAR_pred_input_func = tf.estimator.inputs.pandas_input_fn(
              x=data_row,
              batch_size=1,
              num_epochs=1,
              shuffle=False)
        predictions = self.model.predict(CLEAR_pred_input_func)

        y=np.array([])
        for p in predictions:
           y= np.append(y, p['probabilities'][1])
        self.local_df.iloc[0,0:self.num_features]=data_row.iloc[0,0:self.num_features]
        self.local_df.loc[0,'prediction'] = y
        self.nn_forecast=y[0]
        CLEARExplainer.create_neighbourhood(self)
        if CLEAR_settings.apply_counterfactual_weights:
            CLEARExplainer.add_counterfactual_rows(self)
            if self.num_counterf > 0:
                CLEARExplainer.adjust_neighbourhood(self,self.neighbour_df.tail(self.num_counterf),\
                                                   CLEAR_settings.counterfactual_weight) 
        CLEARExplainer.perform_regression(self)
        if CLEAR_settings.regression_type=='logistic':
            while  (self.additional_weighting < 6 and \
                   ((self.regression_class != self.nn_class) or \
                   (abs(self.local_prob-self.nn_forecast)> 0.01))):                 
                  CLEARExplainer.adjust_neighbourhood(self,self.neighbour_df.iloc[0,:],10) 
                  CLEARExplainer.perform_regression(self)
        return self   

    def adjust_neighbourhood(self, target_rows, num_copies):
        #add num_copies more observations
        self.additional_weighting += 1    
        temp= pd.DataFrame(columns=self.neighbour_df.columns) 
        temp = temp.append(target_rows, ignore_index= True)
        temp2 =self.neighbour_df.copy(deep=True)
        for k in range(1,num_copies):
            temp = temp.append(target_rows, ignore_index= True)  
        temp3 = temp2.append(temp, ignore_index=True)  
        temp3 = temp3.reset_index(drop=True)
        self.neighbour_df= temp3.copy(deep= True)
        if CLEAR_settings.generate_regression_files == True:
            filename1 = CLEAR_settings.CLEAR_path +'local_'+datetime.now().strftime("%Y%m%d-%H%M%S%f")+'.csv'
            self.neighbour_df.to_csv(filename1)              
        return self
                

    def create_neighbourhood(self):
# =============================================================================
#     Generates a Neighbourhood Dataset for each observation and then performs stepwise regressions.
#     The regressions can be polynomail and can also include interaction terms
#     The regressions can either be either multiple or logistic regressions and
#     can be scored using AIC, adjusted R-squared or McFadden's pseudo R-squared
# =============================================================================    
 
# NEED TO REMOVE EXPERIMENT        
        if CLEAR_settings.neighbourhood_algorithm == 'Balanced':    
            if (self.local_df.loc[0,'prediction']>=0.1) & (self.local_df.loc[0,'prediction']<=0.9):
                neighbour_pt1=0.1
                neighbour_pt2 = 0.4
                neighbour_pt3 = 0.6
                neighbour_pt4 = 0.9
            else:
                neighbour_pt1=0
                neighbour_pt2 = 0.4
                neighbour_pt3 = 0.6
                neighbour_pt4 = 1    
            self.local_df.loc[self.local_df['prediction'].between(neighbour_pt1, neighbour_pt2, inclusive = True),'target_range']=1
            self.local_df.loc[self.local_df['prediction'].between(neighbour_pt2, neighbour_pt3, inclusive = True),'target_range']=2
            self.local_df.loc[self.local_df['prediction'].between(neighbour_pt3, neighbour_pt4, inclusive = True),'target_range']=3           
            distances = sklearn.metrics.pairwise_distances(
                    self.local_df.iloc[:,0:self.num_features].values,
                    self.local_df.iloc[0,0:self.num_features].values.reshape(1,-1),
                    metric='euclidean'
            ).ravel()            
            self.local_df['distances']= distances
            self.local_df= self.local_df.sort_values(by=['distances'])
            self.local_df = self.local_df.reset_index(drop=True)
            num_rows = CLEAR_settings.regression_sample_size *(neighbour_pt2/(neighbour_pt4-neighbour_pt1))
            temp_df=self.local_df[self.local_df['target_range']==1]
            temp_df= temp_df.sort_values(by=['distances'])
            temp_df = temp_df.iloc[0:int(num_rows),:]
            self.neighbour_df=temp_df.copy(deep= True)
            num_rows = int(CLEAR_settings.regression_sample_size *(neighbour_pt3-neighbour_pt2)/(neighbour_pt4-neighbour_pt1))
            temp_df=self.local_df[self.local_df['target_range']==2]
            temp_df= temp_df.sort_values(by=['distances'])
            temp_df = temp_df.iloc[0:int(num_rows),:]
            self.neighbour_df=self.neighbour_df.append(temp_df,ignore_index=True)
            num_rows = int(CLEAR_settings.regression_sample_size *(neighbour_pt4-neighbour_pt3)/(neighbour_pt4-neighbour_pt1))
            temp_df=self.local_df[self.local_df['target_range']==3]
            temp_df= temp_df.sort_values(by=['distances'])
            temp_df = temp_df.iloc[0:int(num_rows),:]
            self.neighbour_df=self.neighbour_df.append(temp_df,ignore_index=True)  
            self.neighbour_df= self.neighbour_df.sort_values(by=['distances'])
            self.neighbour_df = self.neighbour_df.reset_index(drop=True)
            if CLEAR_settings.generate_regression_files == True:
                filename1 = CLEAR_settings.CLEAR_path +'local_'+str(self.observation_num) +'_' + datetime.now().strftime("%Y%m%d-%H%M%S%f")+'.csv'
                self.neighbour_df.to_csv(filename1)   
#Creates L1 neighbourhood.selects s observations of synthetic data that are
# nearest to the observatiom. It then checks that both classification classes are 
#sufficiently represented        
        elif CLEAR_settings.neighbourhood_algorithm == 'Unbalanced':
            distances = sklearn.metrics.pairwise_distances(
                    self.local_df.iloc[:,0:self.num_features].values,
                    self.local_df.iloc[0,0:self.num_features].values.reshape(1,-1),
                    metric='euclidean'
            ).ravel()            
            self.local_df['distances']= distances
            self.local_df= self.local_df.sort_values(by=['distances'])
            self.local_df = self.local_df.reset_index(drop=True)
            temp_df = self.local_df.iloc[0:int(200),:]
            self.neighbour_df=temp_df.copy(deep= True)
            if CLEAR_settings.generate_regression_files == True:
                filename1 = CLEAR_settings.CLEAR_path +'local_'+str(self.observation_num) +'_' + datetime.now().strftime("%Y%m%d-%H%M%S%f")+'.csv'
                self.neighbour_df.to_csv(filename1)       
        else:
          print('Neighbourhood Algorithm Misspecified')
        return(self)


    def add_counterfactual_rows(self):
        self.counterf_rows_df= pd.DataFrame(columns=self.neighbour_df.columns) 
        for feature in self.neighbour_df.columns[0:-3]:   
              c_df=self.sensitivity_df[(self.sensitivity_df['observation']==self.observation_num) & (self.sensitivity_df['feature']==feature)]
              c_df=c_df['probability'].agg(['min','max'])
              if (c_df['min']<=0.5) & (c_df['max']>0.5):           
                old_value = self.neighbour_df.loc[0,feature]
                perc50=CLEAR_perturbations.Get_Counterfactual(self.sensitivity_df,feature, old_value, self.observation_num)
                # This is necessary for cases where the observation in sensitivity_df nearest to 50th percentile
                # is the last observation and hence is not identified by Get_Counterfactual.
                if np.isnan(perc50):
                    continue
    #estimate new_value corresponding to 50th percentile. This Gridsearch assumes only a single 50th percentile           
                s1=self.neighbour_df.iloc[0].copy(deep=True)
                s2=pd.Series(s1)           
                s2['target_range']='counterf'
                s2['distances']=np.nan
                s2.loc[feature]= perc50
                self.counterf_rows_df=self.counterf_rows_df.append(s2,ignore_index=True) 
        self.num_counterf= self.counterf_rows_df.shape[0]
        if not self.counterf_rows_df.empty:
            CLEAR_pred_input_func = tf.estimator.inputs.pandas_input_fn(
                  x=self.counterf_rows_df.iloc[:,0:-3],
                  batch_size=1,
                  num_epochs=1,
                  shuffle=False)
            predictions = self.model.predict(LIME_pred_input_func)    
            y=np.array([])
            for p in predictions:
               y= np.append(y, p['probabilities'][1])
            self.counterf_rows_df['prediction']= y                
            self.neighbour_df=self.neighbour_df.append(self.counterf_rows_df,ignore_index=True)
        return(self)




            
    def perform_regression(self):    
#transform neighbourhood data so that it passes through the data point to be explained
        X = self.neighbour_df.iloc[:,0:self.num_features].copy(deep=True)
        X = X.reset_index(drop=True)
        if CLEAR_settings.regression_type in ['logistic','multiple']:
            decision_threshold = 0.5
            if CLEAR_settings.with_indicator_feature == True:
                indicatorFeature_value = np.where((X[CLEAR_settings.feature_with_indicator]>=CLEAR_settings.indicator_threshold), 1, 0)
                X.insert(1,'IndicatorFeature',indicatorFeature_value)
            #Take out features that are redundent or of little predictive power
            if CLEAR_settings.case_study =='Credit Card':
                X.drop(['marDd3','marDd1','eduDd0','eduDd1','eduDd2','eduDd3','eduDd4','eduDd5','eduDd6','genDd1'], axis=1, inplace=True)                
            if CLEAR_settings.no_polynomimals ==True:
                poly_df = X.copy(deep=True)
            else:
                if CLEAR_settings.interactions_only ==True:
                    poly = PolynomialFeatures(interaction_only = True) 
    
                else:    
                    poly = PolynomialFeatures(2)            
                all_poss= poly.fit_transform(X) 
                poly_names = poly.get_feature_names(X.columns)
                poly_names = [w.replace('^2', '_sqrd') for w in poly_names]
                poly_names = [w.replace(' ', '_') for w in poly_names]  
                poly_df = pd.DataFrame(all_poss, columns=poly_names) 
            poly_df_org_first_row=poly_df.iloc[0,:]
            org_poly_df = poly_df.copy(deep=True)
#NOw transform so that regression goes through the data point to be explained            
            if CLEAR_settings.regression_type=='multiple' and CLEAR_settings.no_centering==False:
                Y= self.neighbour_df.loc[:,'prediction'] - self.nn_forecast 
                poly_df= poly_df-poly_df.iloc[0,:]             
            else:
                Y= self.neighbour_df.loc[:,'prediction'].copy(deep=True)
            
            Y = Y.reset_index(drop=True)       
            Y_cont = Y.copy(deep=True)                          
#stepwise regression's choice of variables is restricted, but this was found to improve fidelity.            
            if CLEAR_settings.case_study =='PIMA Indians Diabetes':
                #selected =['1', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
                selected =['1', 'BloodPressure','SkinThickness','BMI','Pregnancies', 'Glucose',  'Insulin', 'DiabetesPedigreeFunction', 'Age']
                remaining = poly_df.columns.tolist()
                try:
                    for x in selected:
                        remaining.remove(x)  
                except:
                    pass
            if CLEAR_settings.case_study =='BreastC':
                #selected =['1', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
#                selected =['1', 'radius','texture','perimeter','area','smoothness','compactness','concavity',\
#                           'concavepoints','symmetry','fractaldimension']
                selected = ['1']
                remaining = poly_df.columns.tolist()
                try:    
                     for x in selected:
                         remaining.remove(x)          
                except:
                    pass
            elif CLEAR_settings.case_study =='Census':
                educ=['eduDdBachelors','eduDdCommunityCollege','eduDdDoctorate','eduDdHighGrad','eduDdMasters','eduDdProfSchool']
                non_null_educ = [col for col in educ if not (X.loc[:, col] == 0).all()]
                non_null_educ = [col for col in educ if (X.loc[:, col].sum() >=10)]
                selected=['1', 'age','hoursPerWeek']
                selected = selected + non_null_educ                 
                non_null_columns = [col for col in poly_df.columns[3:] if ((poly_df.loc[:, col].min() == 0) &(poly_df.loc[:, col].sum() < 10))]
                poly_df.drop(non_null_columns, axis=1, inplace=True)   
                remaining = poly_df.columns.tolist()
                for x in remaining:
                    if x.endswith('_sqrd'):
                      if x not in ['age_sqrd','hoursPerWeek']:
                         poly_df.drop(x, axis=1, inplace=True) 
                remaining = poly_df.columns.tolist()         
            elif CLEAR_settings.case_study =='Credit Card':
                selected=['1', 'LIMITBAL','AGE','PAY0','PAY6','BILLAMT1','BILLAMT6', 'PAYAMT1','PAYAMT6']
#LIME                selected= ['1','PAY0', 'PAYAMT1', 'PAYAMT2', 'marDd2', 'BILLAMT1', 'PAYAMT5', 'LIMITBAL', 'PAYAMT3', 'BILLAMT6', 'PAY2', 'PAY4', 'BILLAMT2', 'PAY5', 'genDd1', 'BILLAMT3', 'BILLAMT4', 'PAYAMT4', 'PAYAMT6']
                for x in self.feature_list:
                  temp_df=self.sensitivity_df[(self.sensitivity_df['observation']==self.observation_num) & (self.sensitivity_df['feature']==x)]
                  temp_df=temp_df['probability'].agg(['min','max'])
                  if (temp_df['min']<=0.5) & (temp_df['max']>0.5):    
                        if not x in selected:
                            selected.append(x)           
                #non_null_columns = [col for col in poly_df.columns if ((poly_df.loc[:, col].min() == 0) &(poly_df.loc[:, col].sum() < 10))]
                
                
                
                remaining = poly_df.columns.tolist()
                for x in remaining:
                    if x.endswith('_sqrd'):
                      if x.startswith(('mar','edu','gen','Indic')):
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
             # adapted from http://trevor-smith.github.io/stepwise-post/
            current_score, best_new_score = -1000, -1000
            while remaining and current_score == best_new_score and len(selected)<CLEAR_settings.max_predictors:
                scores_with_candidates = []
                for candidate in remaining:
                    if CLEAR_settings.regression_type=='multiple' and CLEAR_settings.no_centering == False:
                        formula = "{} ~ {}".format('prediction',' + '.join(selected+ [candidate])+'-1')
                    elif CLEAR_settings.regression_type=='multiple' and CLEAR_settings.no_centering == True:
                        formula = "{} ~ {} ".format('prediction', ' + '.join(selected + [candidate]))
                    else:
                        formula = "{} ~ {}".format('prediction',' + '.join(selected+ [candidate]))
#                    with open("Formula.txt", "a") as file:
#                        file.write(formula + '\n')
                    try:
                        if CLEAR_settings.score_type == 'aic':
                            if CLEAR_settings.regression_type=='multiple':
                                score = sm.GLS.from_formula(formula, poly_df).fit(disp=0).aic
                            else:
                                score = sm.Logit.from_formula(formula, poly_df).fit(disp=0).aic 
                            score = score *-1
                        elif CLEAR_settings.score_type == 'prsquared':
                            if CLEAR_settings.regression_type=='multiple':
                                print("Error prsquared is not used with multiple regression")
                                exit
                            else:
                                score = sm.Logit.from_formula(formula, poly_df).fit(disp=0).prsquared
                        elif CLEAR_settings.score_type == 'adjR':
                            if CLEAR_settings.regression_type=='multiple':
                                score = sm.GLS.from_formula(formula, poly_df).fit(disp=0).rsquared_adj
                                #                                exit
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
                        
                        #error_flag = 1
                        #current_score = 1
                        #break
                if len(scores_with_candidates)>0:
                    scores_with_candidates.sort()    
                    best_new_score, best_candidate = scores_with_candidates.pop()
                    #if current_score < best_new_score:
                    if current_score < best_new_score:    
                        remaining.remove(best_candidate)
                        selected.append(best_candidate)
                        current_score = best_new_score
                    else:
                        break
                    #perfect separation
                else:
                    break
            if CLEAR_settings.regression_type=='multiple' and CLEAR_settings.no_centering == False:
                formula = "{} ~ {}".format('prediction',' + '.join(selected)+'-1')
                selected.remove('1')
            else:
                formula = "{} ~ {}".format('prediction',' + '.join(selected))
            try:
                if CLEAR_settings.regression_type=='logistic':
                    classifier = sm.Logit.from_formula(formula, poly_df).fit(disp=0)
                else:
                    classifier = sm.GLS.from_formula(formula, poly_df).fit(disp=0)
                if CLEAR_settings.score_type == 'aic':
                    self.prediction_score = classifier.aic
                elif CLEAR_settings.score_type == 'prsquared':
                    self.prediction_score = classifier.prsquared
                elif  CLEAR_settings.score_type == 'adjR':  
                    self.prediction_score = classifier.rsquared_adj
                else:
                    print('incorrect score type')
                predictions=classifier.predict(poly_df)               
                self.features= selected
                self.coeffs=classifier.params.values
                self.standard_error=classifier.bse.values
                self.z_scores = classifier.tvalues.values
                self.p_values =classifier.pvalues.values
                #local prob is for the target point is in class 0 . CONFIRM!
                self.local_prob = classifier.predict(poly_df)[0]
                if CLEAR_settings.regression_type== 'logistic':
                    self.accuracy = (classifier.pred_table()[0][0]
                    +classifier.pred_table()[1][1])/classifier.pred_table().sum()
                else:
                    Z=Y.copy(deep=True)
                    Z[Z >= 0.5] = 2
                    Z[Z< 0.5] = 1
                    Z[Z==2] = 0
                    W=predictions.copy(deep=True)
                    W[W >= 0.5] = 2
                    W[W< 0.5] = 1
                    W[W==2] = 0
                    self.accuracy=(W==Z).sum()/Z.shape[0]     
                if self.local_prob>=decision_threshold:
                    self.regression_class = 1
                else:
                    self.regression_class = 0
                if self.nn_forecast>=decision_threshold:
                    self.nn_class = 1
                else:
                    self.nn_class = 0                


                if CLEAR_settings.regression_type=='logistic' or \
                   (CLEAR_settings.regression_type=='multiple' and CLEAR_settings.no_centering == True):
                   self.intercept=classifier.params[0] 
                   self.local_data = []  
                   for i in range(len(selected)):
                       selected_feature= selected[i]
                       for j in range(len(classifier.params)):
                           coeff_feature = classifier.params.index[j]
                           if selected_feature==coeff_feature:
                              self.local_data.append(poly_df_org_first_row.loc[selected_feature])
                   self.untransformed_predictions = classifier.predict(org_poly_df)                
                else:
                    self.local_data = []    
                    temp = 0    
                    self.intercept = +self.nn_forecast
                    for i in range(len(selected)):
                        selected_feature= selected[i]
                        for j in range(len(classifier.params)):
                            coeff_feature = classifier.params.index[j]
                            if selected_feature==coeff_feature:
                                self.intercept -= poly_df_org_first_row.loc[selected_feature]*classifier.params[j]
                                temp  -= poly_df_org_first_row.loc[selected_feature]*classifier.params[j]
                                self.local_data.append(poly_df_org_first_row.loc[selected_feature])
                                adjustment= self.nn_forecast-classifier.predict(poly_df_org_first_row)
                                self.untransformed_predictions =  adjustment[0] +classifier.predict(org_poly_df)     
            except:
                print(formula)
#                input("Regression failed. Press Enter to continue...")  
           
            

            
        else:
            print('incorrect regression type specified')
            exit
        return self
    

                

  
def Run_Regressions(X_test_sample,explainer,feature_list):    
    """ Creates an object of class CLEARExplainer. This creates the Synthetic Dataset
        and calculates some statistics for the variables in X_train
    """    
   
    """  Creates an object of class CLEARExplainer.explain_data_point which performs
         the stepwise regressions. The results of the stepwise regression are
         stored in the results_df dataframe
    """     
    results_df = pd.DataFrame(columns=['Reg_Score','intercept', 'features','weights',\
                                       'nn_forecast','reg_prob','regression_class',\
                                       'spreadsheet_data','local_data','accuracy'])
    observation_num = CLEAR_settings.first_obs       
    print('Performing step-wise regressions \n')
    for i in range(CLEAR_settings.first_obs,CLEAR_settings.last_obs+1):
        data_row=pd.DataFrame(columns=feature_list)
        data_row=data_row.append(X_test_sample.iloc[i],ignore_index=True)
        data_row.fillna(0, inplace= True)
        regression_obj = explainer.explain_data_point(data_row, observation_num)
        print('Processed observation ' + str(i))
        results_df.at[i,'features'] = regression_obj.features
        results_df.loc[i,'Reg_Score'] = regression_obj.prediction_score
        results_df.loc[i,'nn_forecast'] = regression_obj.nn_forecast
        results_df.loc[i,'reg_prob'] = regression_obj.local_prob  
        results_df.loc[i,'regression_class'] = regression_obj.regression_class
        results_df.at[i,'spreadsheet_data'] =regression_obj.local_data
        results_df.at[i,'local_data'] =data_row.values[0]
        results_df.loc[i,'accuracy'] = regression_obj.accuracy
        results_df.loc[i,'intercept'] = regression_obj.intercept
        results_df.at[i,'weights'] = regression_obj.coeffs

        observation_num += 1  
    filename1 = CLEAR_settings.CLEAR_path +'CLRreg_'+ datetime.now().strftime("%Y%m%d-%H%M")+'.csv'   
#    filename2 = CLEAR_settings.CLEAR_path +'Results_'+ datetime.now().strftime("%Y%m%d-%H%M")+'.pkl'  
    results_df.to_csv(filename1)
#    results_df.to_pickle(filename2) 
    return(results_df, regression_obj)



             
            
        
