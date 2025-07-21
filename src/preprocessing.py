'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

# import the necessary packages
import pandas as pd

def preprocessing():
    # load the datasets
    pred_universe = pd.read_csv("data/pred_universe_raw.csv", parse_dates=["arrest_date_univ"])
    arrest_events = pd.read_csv("data/arrest_events_raw.csv", parse_dates=["arrest_date_event"])

    # full outer join on 'person_id' to df_arrest
    df_arrests = pd.merge(pred_universe, arrest_events, on='person_id', how='outer')

    
    # Create column `y`: rearrested for a felony within 1 year
    df_arrests['y'] = 0
    for index, row in df_arrests.iterrows():    
        arrest_date = row['arrest_date_univ']
        person_id = row['person_id']
        if pd.notna(arrest_date):
            felony_arrests = arrest_events[
                (arrest_events['person_id'] == person_id) &
                (arrest_events['arrest_date_event'] > arrest_date) &
                (arrest_events['arrest_date_event'] <= arrest_date + pd.Timedelta(days=365)) &
                (arrest_events['charge_degree'] == 'felony')
            ]
            if not felony_arrests.empty:
                df_arrests.at[index, 'y'] = 1

    print("What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?")
    print(f"Answer: {df_arrests['y'].mean():.2%}")

    # Create feature: current_charge_felony
    df_arrests['current_charge_felony'] = (df_arrests['charge_degree'] == 'felony').astype(int)
    print("What share of current charges are felonies?")
    print(f"Answer: {df_arrests['current_charge_felony'].mean():.2%}")

    # Create feature: num_fel_arrests_last_year
    df_arrests['num_fel_arrests_last_year'] = 0
    for index, row in df_arrests.iterrows():
        arrest_date = row['arrest_date_univ']
        person_id = row['person_id']
        if pd.notna(arrest_date):
            prior_felonies = arrest_events[
                (arrest_events['person_id'] == person_id) &
                (arrest_events['arrest_date_event'] >= arrest_date - pd.Timedelta(days=365)) &
                (arrest_events['arrest_date_event'] < arrest_date) &
                (arrest_events['charge_degree'] == 'felony')
            ]
            df_arrests.at[index, 'num_fel_arrests_last_year'] = len(prior_felonies)

    # Merge num_fel_arrests_last_year into pred_universe
    pred_universe = pred_universe.merge(
        df_arrests[['person_id', 'num_fel_arrests_last_year']],
        on='person_id',
        how='left'
    )

    # Print final outputs
    print("Mean of 'num_fel_arrests_last_year':", pred_universe['num_fel_arrests_last_year'].mean())
    print(pred_universe.head())
    # Return df_arrests for use in main.py
    return df_arrests






