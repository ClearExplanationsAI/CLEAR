from tkinter import *
import sys

""" Specifes CLEAR'S user input parameters. CLEAR sets the input parameters as global variables
whose values are NOT changed in any other module (these are CLEAR's only global variables).
Tkinter is used to provide some checks on the user's inputs. The file 
'Input Parameters for CLEAR.pdf' on Github documents the input parameters.
 """
def init():
    global case_study, max_predictors, first_obs,last_obs, num_samples, regression_type,\
    score_type, test_sample, regression_sample_size, min_target_sample,\
    feature_with_indicator,CLEAR_path, with_indicator_feature, indicator_threshold,\
    neighbourhood_algorithm,perturb_one_feature,only_feature_perturbed,\
    apply_counterfactual_weights,counterfactual_weight, Num_iterations,\
    generate_regression_files,LIME_comparison,interactions_only, no_centering,no_polynomimals,\
    LIME_sample, LIME_kernel
    
    case_study = 'PIMA Indians Diabetes' # 'Credit Card','PIMA Indians Diabetes','Credit Card','BreastC'
    max_predictors = 11  # maximum number of dependent variables in stepwise regression
    first_obs =1
    last_obs=1  # number of observations to analyse in CLEAR test dataset Census 115/225 in test1 PIMA 115 in test1
    # Credit 104
    num_samples=1000 # number of observations to generate in Synthetic Dataset. Defaul 100000
    regression_type = 'logistic' #'multiple' 'logistic'
    score_type = 'prsquared' # prsquared is McFadden Pseudo R-squared. Can also be
    #                          set to aic or adjR (adjusted R-squared)
    test_sample=1            # sets CLEAR's test dataset
    regression_sample_size =200   # minimum number of observations in local regression. Default 200
    feature_with_indicator = 'Glucose'  #age,Glucose
    CLEAR_path='D:/CLEAR/' #'D:/CLEAR/''/content/drive/My Drive/Colab/'
    with_indicator_feature = False    # whether to use this indicator variable
    indicator_threshold = 1.5  # threshold for indicator variable # for PAY 0 0=0.1, 1 =0.91
    neighbourhood_algorithm= 'Balanced' #default is L3 . Tested against Unbalanced
    perturb_one_feature = False # perturb only one feature eg 'age'
    only_feature_perturbed = 'age' # the single feature that is perturbed if
                                  # 'perturb_one_feature' = True 
    apply_counterfactual_weights = False
    counterfactual_weight = 9 # default to 9
    generate_regression_files = False
    Num_iterations= 1
# Parameters for evaluating the effects of different parts of CLEAR's regression
    interactions_only = False
    no_centering = False
    no_polynomimals = False                             
# Parameters for comparing CLEAR with LIME
    LIME_comparison = False
    LIME_sample = 15000  # number of synthetic data-points generated
    LIME_kernel = 0.75  # LIME kernel width. Set to None to use Ribeiro et al.'s default formula


    #check for inconsistent input data
    check_input_parameters()

""" Check if input parameters are consistent"""
def check_input_parameters():

    def close_program(): 
            root.destroy()
            sys.exit()
    
    error_msg = ""   
    if perturb_one_feature == True and case_study != 'Census':
        error_msg = "'Perturb_one_feature' only applies to 'Census' dataset"
    elif first_obs > last_obs:
        error_msg = "last_obs must be greater or equal to first obs"
    elif last_obs> 100:
        error_msg = "Neurips case studies were carried out on first 100 observations"
    elif perturb_one_feature == False and case_study == 'Census':
        error_msg = "'Perturb_one_feature' currently needs to apply to 'Census' dataset"        
    elif perturb_one_feature == True and only_feature_perturbed  != 'age':
        error_msg = "'Only feature perturbed' should be set to age"    
    elif regression_type == 'logistic' and \
        (score_type != 'prsquared' and score_type != 'aic'):
        error_msg = "logistic regression and score type combination incorrectly specified"
    elif regression_type == 'multiple' and   score_type == 'prsquared':
        error_msg = "McFadden Pseudo R-squared cannot be used with multiple regression"
    elif case_study not in ['Census','PIMA Indians Diabetes','Credit Card','BreastC'] :
        error_msg = "Case study incorrectly specified"
    elif regression_type not in ['multiple', 'logistic']:
        error_msg = "Regression type misspecified"  
    elif neighbourhood_algorithm not in ['Balanced', 'Unbalanced']:
        error_msg = "neighbourhood algorithm misspecified"       
    elif LIME_comparison == True and  regression_type != 'multiple' :
        error_msg = "LIME comparison only works with multiple regression"                
    elif (isinstance((interactions_only & no_centering & no_polynomimals & \
                     apply_counterfactual_weights & generate_regression_files & \
                     LIME_comparison),bool))is False:
        error_msg = "A boolean variable has been incorrectly specified"                
 
    
    if  error_msg != "":    
        root=Tk()
        root.title("Input Error in CLEAR_settings")
        root.geometry("350x150")

        label_1 = Label(root,text =error_msg, \
                        justify = CENTER, height = 4, wraplength = 150)
        button_1 = Button(root, text= "OK", \
                          padx = 5, pady = 5, command = close_program)
        label_1.pack()
        button_1.pack()
        root.attributes("-topmost", True)
        root.focus_force() 
        root.mainloop()
    

#    