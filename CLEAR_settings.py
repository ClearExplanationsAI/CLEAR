from tkinter import *

""" Specifes CLEAR'S user input parameters. CLEAR sets the input parameters as global variables
whose values are NOT changed in any other module (these are CLEAR's only global variables).
Tkinter is used to provide some checks on the user's inputs. The file 
'Input Parameters for CLEAR.pdf' on Github documents the input parameters.
 """


def init():
    global case_study, max_predictors, first_obs, last_obs, num_samples, regression_type, \
        score_type, test_sample, regression_sample_size, \
        feature_with_indicator, CLEAR_path, with_indicator_feature, indicator_threshold, \
        neighbourhood_algorithm, perturb_one_feature, only_feature_perturbed, \
        apply_counterfactual_weights, counterfactual_weight, num_iterations, \
        generate_regression_files, LIME_comparison, interactions_only, no_centering, no_polynomimals, \
        LIME_sample, LIME_kernel, multi_class, multi_class_labels, multi_class_focus,\
        use_prev_sensitivity, use_sklearn

    case_study = 'PIMA'  # 'Credit Card','PIMA','Census','BreastC','IRIS'
    max_predictors = 15 # maximum number of dependent variables in stepwise regression
    first_obs = 1
    last_obs = 2 # number of observations to analyse in CLEAR test dataset Census 115/225 in test1 PIMA 115 in test1
    # Credit 104
    num_samples = 5000  # number of observations to generate in Synthetic Dataset. Default 50000
    regression_type = 'multiple'  # 'multiple' 'logistic'
    score_type = 'adjR'  # prsquared is McFadden Pseudo R-squared. Can also be
    #                          set to aic or adjR (adjusted R-squared)
    test_sample = 1  # sets CLEAR's test dataset
    regression_sample_size = 200  # minimum number of observations in local regression. Default 200
    feature_with_indicator = 'Glucose'  # age,Glucose
    CLEAR_path = 'D:/CLEAR/'  # 'D:/CLEAR/''/content/drive/My Drive/Colab/'
    with_indicator_feature = False  # whether to use this indicator variable
    indicator_threshold = 1.5  # threshold for indicator variable # for PAY 0 0=0.1, 1 =0.91
    neighbourhood_algorithm = 'Balanced'  # default is L3 . Tested against Unbalanced
    perturb_one_feature = False  # perturb only one feature eg 'age'
    only_feature_perturbed = 'age'  # the single feature that is perturbed if
    # 'perturb_one_feature' = True
    apply_counterfactual_weights = False
    counterfactual_weight = 9  # default to 9-
    generate_regression_files = False
    num_iterations = 1
    # Parameters for evaluating the effects of different parts of CLEAR's regression
    interactions_only = False
    no_centering = False
    no_polynomimals = False
    # Parameters for comparing CLEAR with LIME
    LIME_comparison = True
    LIME_sample = 5000  # number of synthetic data-points generated
    LIME_kernel = 2  # LIME kernel width. Set to None to use Ribeiro et al.'s default formula
    # parameters for multi-class datasets
    multi_class = False
    multi_class_labels = ['setosa','versicolor','virginica']
                          # ordering must correspond to the numeric labels
    # of classes in dataset eg ['setosa','versicolor','virginica']  corresponds to setosa = 0, versicolor = 1....
    # [1, 2, 3, 5, 6, 7]
    multi_class_focus = 'setosa'  # eg 'setosa' or 'All'
    use_prev_sensitivity = False
    use_sklearn = False
    # check for inconsistent input data
    check_input_parameters()


""" Check if input parameters are consistent"""


def check_input_parameters():
    def close_program():
        root.destroy()
        sys.exit()

    error_msg = ""
    if perturb_one_feature is True and case_study != 'Census':
        error_msg = "'Perturb_one_feature' only applies to 'Census' dataset"
    elif multi_class == True and ((multi_class_focus not in multi_class_labels) and multi_class_focus != 'All'):
        error_msg = "multi_class focus incorrectly specified"
    elif multi_class == True and case_study not in ['IRIS', 'Glass']:
        error_msg = "multi_class is True but case study is binary"
    elif multi_class == False and case_study in ['IRIS', 'Glass']:
        error_msg = "multi_class is False but case study is a multi-class dataset "
    elif first_obs > last_obs:
        error_msg = "last_obs must be greater or equal to first obs"
    elif last_obs > 100:
        error_msg = "AAAI case studies were carried out on first 100 observations"
    elif perturb_one_feature is False and case_study == 'Census':
        error_msg = "'Perturb_one_feature' currently needs to apply to 'Census' dataset"
    elif perturb_one_feature is True and only_feature_perturbed != 'age':
        error_msg = "'Only feature perturbed' should be set to age"
    elif regression_type == 'logistic' and \
            (score_type != 'prsquared' and score_type != 'aic'):
        error_msg = "logistic regression and score type combination incorrectly specified"
    elif regression_type == 'multiple' and score_type == 'prsquared':
        error_msg = "McFadden Pseudo R-squared cannot be used with multiple regression"
    elif case_study not in ['Census', 'PIMA', 'Credit Card', 'BreastC', 'IRIS', 'Glass']:
        error_msg = "Case study incorrectly specified"
    elif regression_type not in ['multiple', 'logistic']:
        error_msg = "Regression type misspecified"
    elif neighbourhood_algorithm not in ['Balanced', 'Unbalanced']:
        error_msg = "neighbourhood algorithm misspecified"
    elif LIME_comparison is True and regression_type != 'multiple':
        error_msg = "LIME comparison only works with multiple regression"
    elif (isinstance((interactions_only & no_centering & no_polynomimals &
                      apply_counterfactual_weights & generate_regression_files &
                      LIME_comparison), bool)) is False:
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

#
