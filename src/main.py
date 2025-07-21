'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import etl
import preprocessing
import logistic_regression
import decision_tree
import calibration_plot


# Call functions / instanciate objects from the .py files
def main():

    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    etl.etl()
    print("ETL completed. Data saved in `data/` directory.")
    # print(pred_universe_raw.head()) 
    # print(arrest_events_raw.head())

    # PART 2: Call functions/instanciate objects from preprocessing
    df_arrests = preprocessing.preprocessing()
    print("Preprocessing completed.")
    # print(df_arrests.head())

    # PART 3: Call functions/instanciate objects from logistic_regression
    df_arrests_train, df_arrests_test, = logistic_regression.logistic_regression(df_arrests)

    # PART 4: Call functions/instanciate objects from decision_tree
    df_arrests_test = decision_tree.decision_tree(df_arrests_train, df_arrests_test)

    # PART 5: Call functions/instanciate objects from calibration_plot
    calibration_plot.calibration_plot_main(df_arrests_test)


if __name__ == "__main__":
    main()


