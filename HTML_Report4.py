import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.formula.api as sm 
import CLEAR_settings
from jinja2 import Environment, FileSystemLoader
from math import log10, floor

observation_num=1
num_features = 8
feature_list=['BloodPressure','SkinThickness','BMI','Pregnancies', 'Glucose',  'Insulin', 'DiabetesPedigreeFunction', 'Age']

CLEAR_settings.init()

if CLEAR_settings.regression_type== 'logistic':
    nncomp_df=pd.read_csv(CLEAR_settings.CLEAR_path +'log_report_nncomp.csv')
    neighbour_df = pd.read_csv(CLEAR_settings.CLEAR_path +'log_neighbour.csv')
else:
    nncomp_df=pd.read_csv(CLEAR_settings.CLEAR_path +'mult_report_nncomp.csv')
    neighbour_df = pd.read_csv(CLEAR_settings.CLEAR_path +'mult_neighbour.csv')
    
sensitivity_df = pd.read_csv(CLEAR_settings.CLEAR_path +'Final_diabetes_sensitivity1.csv')  
data_row= neighbour_df.iloc[0,0:num_features+1]
#neighbour_df.prediction= np.log(neighbour_df.prediction) 
original_prediction=neighbour_df.loc[0,'prediction']
X = neighbour_df.iloc[:,0:num_features].copy(deep=True)
X = X.reset_index(drop=True)
 
if CLEAR_settings.regression_type in ['logistic','multiple']:
    decision_threshold = 0.5
    if CLEAR_settings.with_indicator_feature == True:
        indicatorFeature_value = np.where((X[CLEAR_settings.feature_with_indicator]>=CLEAR_settings.indicator_threshold), 1, 0)
        X.insert(1,'IndicatorFeature',indicatorFeature_value)
    poly = PolynomialFeatures(2)            
#            use poly = PolynomialFeatures(interaction_only = True) if only interactions required
    all_poss= poly.fit_transform(X) 
    poly_names = poly.get_feature_names(X.columns)
    poly_names = [w.replace('^2', '_sqrd') for w in poly_names]
    poly_names = [w.replace(' ', '_') for w in poly_names]
    poly_df = pd.DataFrame(all_poss, columns=poly_names) 
    poly_df_org_first_row=poly_df.iloc[0,:] 
    org_poly_df = poly_df.copy(deep=True)
#NOw transform so that regression goes through the data point to be explained            
    if CLEAR_settings.regression_type=='multiple':
        Y= neighbour_df.loc[:,'prediction'] - original_prediction 
        poly_df= poly_df-poly_df.iloc[0,:]             
    else:
        Y= neighbour_df.loc[:,'prediction'].copy(deep=True)
    
    Y = Y.reset_index(drop=True)       
    Y_cont = Y.copy(deep=True)                          
#stepwise regression's choice of variables is restricted, but this was found to improve fidelity.            
    if CLEAR_settings.case_study =='PIMA Indians Diabetes':
        #selected =['1', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        selected =['1', 'BloodPressure','SkinThickness','BMI','Pregnancies', 'Glucose',  'Insulin', 'DiabetesPedigreeFunction', 'Age']
        remaining = poly_df.columns.tolist()
        for x in selected:
            remaining.remove(x)     
    elif CLEAR_settings.case_study =='Census':
    #educ=['eduDdBachelors','eduDdCommunityCollege','eduDdDoctorate','eduDdHighGrad','eduDdMasters','eduDdProfSchool']
#            non_null_educ = [col for col in educ if not (X.loc[:, col] == 0).all()]
#            non_null_educ = [col for col in educ if (X.loc[:, col].sum() >=10)]
        selected=['1', 'age']
#           selected = selected + non_null_educ                 
        non_null_columns = [col for col in poly_df.columns if ((poly_df.loc[:, col].min() == 0) &(poly_df.loc[:, col].sum() < 10))]
        poly_df.drop(non_null_columns, axis=1, inplace=True)   
        remaining = poly_df.columns.tolist()
        for x in remaining:
            if x.endswith('_sqrd'):
              if x not in ['age_sqrd','hoursPerWeek']:
                 poly_df.drop(x, axis=1, inplace=True) 
        remaining = poly_df.columns.tolist()         
    elif CLEAR_settings.case_study =='Credit Card':
        #selected=['1', 'LIMITBAL','AGE','PAY0','PAY6','BILLAMT1','BILLAMT6', 'PAYAMT1','PAYAMT6']
        selected=['LIMITBAL','AGE','PAY0','PAY2','PAY3','PAY4','PAY5','PAY6','PAYAMT1','BILLAMT1']
#                selected=['1']
        for x in feature_list:
          temp_df=sensitivity_df[(sensitivity_df['observation']==observation_num) & (sensitivity_df['feature']==x)]
          temp_df=temp_df['probability'].agg(['min','max'])
          if (temp_df['min']<=0.5) & (temp_df['max']>0.5):    
                if not x in selected:
                    selected.append(x)
    
        non_null_columns = [col for col in poly_df.columns if ((poly_df.loc[:, col].min() == 0) &(poly_df.loc[:, col].sum() < 10))]
        non_null_columns.remove('1') 
        poly_df.drop(non_null_columns, axis=1, inplace=True)    
        remaining = poly_df.columns.tolist()
        for x in remaining:
            if x.endswith('_sqrd'):
              if x.startswith(('mar','edu','gen','Indic')):
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
    while remaining and current_score == best_new_score and len(selected)<CLEAR_settings.max_predictors-1:
        scores_with_candidates = []
        for candidate in remaining:
            
            
            if CLEAR_settings.regression_type=='multiple':
                formula = "{} ~ {}".format('prediction',' + '.join(selected+ [candidate])+'-1')
            else:
                formula = "{} ~ {}".format('prediction',' + '.join(selected+ [candidate]))
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
    
    if CLEAR_settings.regression_type=='multiple':
        formula = "{} ~ {}".format('prediction',' + '.join(selected)+'-1')
    else:
        formula = "{} ~ {}".format('prediction',' + '.join(selected))
    try:
        if CLEAR_settings.regression_type=='logistic':
            classifier = sm.Logit.from_formula(formula, poly_df).fit(disp=0)
        else:
            classifier = sm.GLS.from_formula(formula, poly_df).fit(disp=0)
        if CLEAR_settings.score_type == 'aic':
            prediction_score = classifier.aic
        elif CLEAR_settings.score_type == 'prsquared':
            prediction_score = classifier.prsquared
        elif  CLEAR_settings.score_type == 'adjR':  
            prediction_score = classifier.rsquared_adj
        else:
            print('incorrect score type')
        predictions=classifier.predict(poly_df)               
        features= selected
        coeffs=classifier.params.values
        standard_error=classifier.bse.values
        z_scores = classifier.tvalues.values
        p_values =classifier.pvalues.values
        #local prob is for the target point is in class 0 . CONFIRM!
        local_prob = classifier.predict(poly_df)[0]
        if CLEAR_settings.regression_type== 'logistic':
            accuracy = (classifier.pred_table()[0][0]
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
            accuracy=(W==Z).sum()/Z.shape[0]     
        if local_prob>=decision_threshold:
            regression_class = 1
        else:
            regression_class = 0
        nn_forecast= original_prediction
        if nn_forecast>=decision_threshold:
            nn_class = 1
        else:
            nn_class = 0                

#re-calculate coeffs as neighbourhood_df was transformed so as to force regression
# through the data point to be explained
#Y-Y_0=β(X-X_0 )
#Y=β(X-X_0 )+Y_0
#Y=βX-βX_0+Y_0
        if CLEAR_settings.regression_type=='logistic':
            matched = True
            local_data = []            
            intercept=classifier.params[0] 
            for i in range(len(selected)):
                if matched== False:
                    temp=1
                selected_feature= selected[i]
                matched = False
                for j in range(len(classifier.params)):
                    coeff_feature = classifier.params.index[j]
                    if selected_feature==coeff_feature:
                       local_data.append(poly_df_org_first_row.loc[selected_feature])          
                       matched = True

        else:
            temp = 0    
            intercept = +original_prediction
            for i in range(len(selected)):
                selected_feature= selected[i]
                for j in range(len(classifier.params)):
                    coeff_feature = classifier.params.index[j]
                    if selected_feature==coeff_feature:
                        intercept -= poly_df_org_first_row.loc[selected_feature]*classifier.params[j]
                        temp  -= poly_df_org_first_row.loc[selected_feature]*classifier.params[j]
            print('intercept:  ',intercept)
            print('beta_xo:  ', temp)
    except:
        print(formula)
#                input("Regression failed. Press Enter to continue...")  






    adjustment= original_prediction-classifier.predict(poly_df_org_first_row)
    untransformed_predictions =  adjustment[0] +classifier.predict(org_poly_df)

#dataframe to HTML Report
    def round_sig(x, sig=2):
        return round(x, sig-int(floor(log10(abs(x))))-1)

    if CLEAR_settings.regression_type== 'multiple':
        regression_formula = 'prediction = ' +  str(round_sig(intercept))
    else:
        regression_formula = '<font size = "4.5">prediction =  [ 1 + e<sup><b>-w<sup>T</sup>x</sup></b> ]<sup> -1</sup></font size><br><br>' \
         + '<font size = "4.5"><b><i>w</i></b><sup>T</sup><b><i>x</font size></i></b> =  ' +  str(round_sig(intercept))

    for i in range(len(classifier.params)):
        if classifier.params.index[i] == 'Intercept':
            continue
        elif classifier.params[i]<0:
                regression_formula = regression_formula + ' - '+ str(-1*round_sig(classifier.params[i])) + \
                                     ' ' + classifier.params.index[i]          
        else:
                regression_formula = regression_formula + ' + ' + str(round_sig(classifier.params[i])) + \
                                     ' ' + classifier.params.index[i]                                       
    regression_formula= regression_formula.replace("_sqrd"," sqrd")
    regression_formula= regression_formula.replace("_","*")
    report_AI_prediction = str(round_sig(original_prediction))
    if CLEAR_settings.score_type == 'adjR':  
        report_regression_type = "Adjusted R-Squared"
    else:
        report_regression_type = CLEAR_settings.score_type
        
        
    HTML_df= pd.DataFrame(columns=['feature','input value','coeff','abs_coeff'])
    report_selected = [w.replace('_sqrd', ' sqrd') for w in selected]
    report_selected = [w.replace('_', '*') for w in report_selected]
    for i in range(len(classifier.params)):            
        feature =classifier.params.index[i]
        if feature == 'Intercept':
            continue
        else:
            HTML_df.loc[i,'feature']= classifier.params.index[i]
            HTML_df.loc[i,'input value']= poly_df_org_first_row[feature]
            HTML_df.loc[i,'coeff']=classifier.params[i]
            HTML_df.loc[i,'abs_coeff']=abs(classifier.params[i])

        
    HTML_df=HTML_df.sort_values(by=['abs_coeff'], ascending = False)
    HTML_df.drop(['abs_coeff'],axis =1,inplace=True)
    
    HTML_df=HTML_df.head(10)
     
    

    counter_df =nncomp_df[['feature','old_value','perc50']].copy()
    counter_df.rename(columns={'old_value':'input value', 'perc50':'counterfactual value'}, inplace=True )
#    HTML_df.to_html('CLEAR.HTML')
    
    nncomp_df['error']= nncomp_df['new_value']-nncomp_df['perc50']
    reg_counter_df=nncomp_df[['feature','new_value','error']].copy()
    reg_counter_df.rename(columns={'new_value':'counterfactual value'}, inplace=True )
       
    pd.set_option('colheader_justify', 'left','precision', 2)
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template("CLEAR_report.html")
    template_vars = {"title" : "CLEAR Statistics",
                     "regression_table": HTML_df.to_html(index=False, classes = 'mystyle'),
                     "counterfactual_table": counter_df.to_html(index=False,classes='mystyle'),
#                     "inaccurate_table": inaccurate_df.to_html(index=False,classes='mystyle'),
                     "dataset_name": CLEAR_settings.case_study,
                     "observation_number": observation_num,
                     "regression_formula": regression_formula,
                     "prediction_score": round_sig(prediction_score),
                     "regression_type":report_regression_type,
                     "AI_prediction":report_AI_prediction,
                     "reg_counterfactuals":reg_counter_df.to_html(index=False, classes = 'mystyle')
                     }
    # Render our file and create the PDF using our css style file
    #html_out = template.render(template_vars)
    with open('new_CLEAR.html', 'w') as fh:
        fh.write(template.render(template_vars))





fig = plt.figure()
plt.scatter(neighbour_df.loc[:,'prediction'] ,untransformed_predictions , c='green',s=10)
plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), c= "red", linestyle='-')

plt.xlabel('Target AI System')
if CLEAR_settings.regression_type == 'logistic':
    plt.ylabel('Logistics Regression')
elif  CLEAR_settings.regression_type == 'multiple':   
     plt.ylabel('Multiple Regression')
else:
     plt.ylabel('Polynomial Regression')

fig.savefig('CLEAR_plot.png', bbox_inches = "tight")

plt.show()







