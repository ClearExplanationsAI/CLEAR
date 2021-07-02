from tkinter import *

""" Specifes CLEAR'S user input parameters. CLEAR sets the input parameters as global variables
whose values are NOT changed in any other module (these are CLEAR's only global variables).
Tkinter is used to provide some checks on the user's inputs. The file 
'Input Parameters for CLEAR.pdf' on Github documents the input parameters.
 """


def init():
    global sample_model, max_predictors, first_obs, last_obs, num_samples, regression_type, \
        score_type,logistic_regularise, test_sample, regression_sample_size, CLEAR_path, neighbourhood_algorithm, \
        apply_counterfactual_weights, counterfactual_weight, num_iterations, \
        generate_regression_files, interactions_only, centering, no_polynomimals, multi_class_focus,\
        use_prev_sensitivity, binary_decision_boundary,include_all_numerics,include_features,include_features_list, \
        model_name, numeric_features , categorical_features, category_prefix, class_labels

    model_name = 'Adult'
    numeric_features = ['age', 'hoursPerWeek']
    categorical_features = ['marital_status', 'occupation', 'gender', 'workclass', 'education']
    category_prefix = ['mar', 'occ', 'gen', 'wor', 'edu']
    class_labels = {0: '<=$50K', 1: '> $50K'}
    sample_model = 'Adult'  # This is used by Run_CLEAR_with_sample_model(). Options are
                                # 'Credit Card','PIMA','Adult','BreastC','IRIS'
    max_predictors = 15 # maximum number of dependent variables in stepwise regression
    first_obs = 1  # first observation to analyse from test dataset
    last_obs = 1    # last observation to analyse from test dataset (usually set to < 100)
    num_samples = 50000  # number of observations to generate in Synthetic Dataset. Default 50000
    regression_type = 'logistic'  # 'multiple' 'logistic'
    score_type = 'aic'  # prsquared is McFadden Pseudo R-squared. Can also be
    #                          set to aic or adjR (adjusted R-squared)
    logistic_regularise = False  # it is recommended to keep this as False unless there are problems with perfect separation
    regression_sample_size = 200  # minimum number of observations in local regression. Default 200
    CLEAR_path = 'D:/CLEAR/'  # e.g. 'D:/CLEAR/'
    neighbourhood_algorithm = 'Balanced'  # 'Balanced' 'Unbalanced' Default should be Balanced .
    apply_counterfactual_weights = True
    counterfactual_weight = 9  # default to 9
    generate_regression_files = True
    num_iterations = 1 # number of times that CLEAR will evaluate each observation. Provides the data needed to calc confidence intervals
    # Parameters for evaluating the effects of different parts of CLEAR's regression
    interactions_only = False
    centering = True #forces CLEAR's regression to pass through observation that is to be explained
    no_polynomimals = False
    # Parameters for forcing features to be included in regression
    include_all_numerics = False # This forces the regression model to include all numeric features
    include_features = False # Features in 'include_feature_list' will be forced into regression equation
    include_features_list = [ 'gender']
    # parameters for binary-class datasets
    binary_decision_boundary = 0.5
    # parameters for multi-class datasets.
    multi_class_focus = 'setosa'  # eg 'setosa' or 'All'
    #feature names


    check_input_parameters()
""" Check if input parameters are consistent"""


def check_input_parameters():
    def close_program():
        root.destroy()
        sys.exit()

    error_msg = ""
    if first_obs > last_obs:
        error_msg = "last_obs must be greater or equal to first obs"
    elif regression_type == 'logistic' and \
            (score_type != 'prsquared' and score_type != 'aic'):
        error_msg = "logistic regression and score type combination incorrectly specified"
    elif regression_type == 'multiple' and score_type == 'prsquared':
        error_msg = "McFadden Pseudo R-squared cannot be used with multiple regression"
    elif sample_model not in ['Adult', 'PIMA', 'Credit Card', 'BreastC', 'IRIS', 'German Credit']:
        error_msg = "Sample dataset incorrectly specified"
    elif regression_type not in ['multiple', 'logistic']:
        error_msg = "Regression type misspecified"
    elif neighbourhood_algorithm not in ['Balanced', 'Unbalanced']:
        error_msg = "neighbourhood algorithm misspecified"
    elif (isinstance((interactions_only & centering & no_polynomimals &
                      apply_counterfactual_weights & generate_regression_files), bool)) is False:
        error_msg = "A boolean variable has been incorrectly specified"

    if error_msg != "":
        root = Tk()
        root.title("Input Error in CLEAR_settings")
        root.geometry("350x150")

        label_1 = Label(root, text=error_msg,
                        justify=CENTER, height=4, wraplength=150)
        button_1 = Button(root, text="OK",
                          padx=5, pady=5, command=close_program)
        label_1.pack()
        button_1.pack()
        root.attributes("-topmost", True)
        root.focus_force()
        root.mainloop()

#['LIMITBAL', 'AGE', 'PAY0', 'PAY6', 'BILLAMT1', 'BILLAMT6', 'PAYAMT1', 'PAYAMT6','MARRIAGE']
