'''
PART 5: Calibration-light
Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Which model is more calibrated? Print this question and your answer. 

Extra Credit
Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
Compute AUC for the logistic regression model
Compute AUC for the decision tree model
Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss

# Calibration plot function 
def calibration_plot(y_true, y_prob, n_bins=10):
    """
    Create a calibration plot with a 45-degree dashed line.

    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to divide the data for calibration.

    Returns:
        None
    """
    #Calculate calibration values
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    #Create the Seaborn plot
    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_true, bin_means, marker='o', label="Model")
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()

def calibration_plot_main(df_arrests_test):
    # use the calibration_plot function for logistic regression
    Logistic_Regression = calibration_plot(df_arrests_test['y'], df_arrests_test['pred_lr'], n_bins=5)
    # use the calibration_plot function for decision tree
    Decision_Tree = calibration_plot(df_arrests_test['y'], df_arrests_test['pred_dt'], n_bins=5)
    # Print which model is more calibrated
    print("Which model is more calibrated?")
    brier_lr = brier_score_loss(df_arrests_test['y'], df_arrests_test['pred_lr'])
    brier_dt = brier_score_loss(df_arrests_test['y'], df_arrests_test['pred_dt'])
    print("Brier Score (for more accurate answer)- Logistic Regression:", brier_lr)
    print("Brier Score - Decision Tree:", brier_dt)
    print("Answer: The Decision Tree model is more calibrated.")
    

    ## EXTRA CREDIT ##
    # Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
    top_50_lr = df_arrests_test.nlargest(50, 'pred_lr')
    ppv_lr = top_50_lr['y'].mean()
    # Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
    top_50_dt = df_arrests_test.nlargest(50, 'pred_dt')
    ppv_dt = top_50_dt['y'].mean()
    # Compute AUC for the logistic regression model
    auc_lr = roc_auc_score(df_arrests_test['y'], df_arrests_test['pred_lr'])
    # Compute AUC for the decision tree model
    auc_dt = roc_auc_score(df_arrests_test['y'], df_arrests_test['pred_dt'])
    # Print PPV and AUC results
    print(f"PPV for Logistic Regression (top 50): {ppv_lr:.2%}")
    print(f"PPV for Decision Tree (top 50): {ppv_dt:.2%}")
    print(f"AUC for Logistic Regression: {auc_lr:.2f}")
    print(f"AUC for Decision Tree: {auc_dt:.2f}")
    # Do both metrics agree that one model is more accurate than the other? Print this question and your answer.
    if ppv_lr > ppv_dt and auc_lr > auc_dt:
        print("Both PPV and AUC agree: Logistic Regression is more accurate.")
    elif ppv_lr < ppv_dt and auc_lr < auc_dt:
        print("Both PPV and AUC agree: Decision Tree is more accurate.")
    else:
        print("PPV and AUC disagree on which model is more accurate.")

    