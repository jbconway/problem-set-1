'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: num_fel_arrests_last_year, current_charge_felony
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

# Import any further packages you may need for PART 3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.linear_model import LogisticRegression as lr


# Your code here
def logistic_regression(df_arrests):
    # split the data unto train and test sets
    df_arrests_train, df_arrests_test = train_test_split(df_arrests, test_size=0.3, shuffle=True, stratify=df_arrests['y'])

    # creare a list called features
    features = ['num_fel_arrests_last_year', 'current_charge_felony']

    # create a parameter grid containing three values for the C hyperparameter
    param_grid = {'C': [0.01, 0.1, 1.0]}

    # initialize the Logistic Regression model
    lr_model = lr()

    # initialize the GridSearchCV
    gs_cv = GridSearchCV(estimator=lr_model, param_grid=param_grid, cv=5)

    # run the model
    gs_cv.fit(df_arrests_train[features], df_arrests_train['y'])

    # Print these questions and your answers. 
    print("What was the optimal value for C? Did it have the most or least regularization? Or in the middle?")
    optimal_C = gs_cv.best_params_['C']
    if optimal_C == 0.01:
        print(f"The optimal value for C is {optimal_C}, which has the most regularization.")
    elif optimal_C == 1.0:
        print(f"The optimal value for C is {optimal_C}, which has the least regularization.")
    else:
        print(f"The optimal value for C is {optimal_C}, which is in the middle.")


    # predict for the test set
    df_arrests_test['pred_lr'] = gs_cv.predict(df_arrests_test[features])

    # return the dataframes for use in main.py for PART 4 and PART 5
    return df_arrests_train, df_arrests_test