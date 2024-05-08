import datetime
import json
import plotly.express as px
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

def main():
    # read excel file
    xls = pd.read_excel('data/Eltern-Kind Turnen Freitag 16-17.xlsx', sheet_name=None)
    # xls is a dictionary where the keys are the sheet names and the values are the dataframes
    participation = {}
    all_dates = set()
    all_families = set()
    for sheet_name, df in xls.items():
        print(f"Processing sheet: {sheet_name}")
        # now you can process each dataframe (df)
        for index, row in df.iterrows():
            if index < 2:
                continue

            first_name = row['Vorname/Kind']
            last_name = row['Name']
            full_name = f"{first_name} {last_name}"
            all_families.add(full_name)
            for col in df.columns:
                if type(col) is datetime.datetime:
                    all_dates.add(col)
                    key = f"{col.year}-{str(col.month).zfill(2)}-{str(col.day).zfill(2)}"

                    if row[col] == 1.0:
                        encoded_value = 1
                    else:
                        encoded_value = 0

                    if key in participation:
                        participation[key][full_name] = encoded_value
                    else:
                        participation[key] = {full_name: encoded_value}



    print(json.dumps(participation, indent=4))
    df = pd.DataFrame(participation).T


    # Convert the index to datetime
    df.index = pd.to_datetime(df.index)

    # Fill missing participants that later joined with "0"
    df = df.fillna(0)

    # Drop the families where each date is "0"
    df = df.loc[:, ~(df == 0).all(axis=0)]

    # drop all dates where all families are 0
    df = df.loc[~(df == 0).all(axis=1)]

    # Sort the DataFrame by its index (the dates)
    df = df.sort_index()


    # Sum up the participation of families for each date
    df_sum = df.sum(axis=1)

    # Split the data into training and test sets
    train = df_sum
    test = df_sum[int(0.8*len(df_sum)):]

    # Fit an ARIMA model to the training data
    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit()


    # Use the fitted model to make predictions on the test data
    predictions = model_fit.predict(start=train.shape[0], end=train.shape[0] + 8, dynamic=False)

    # map predcitions to include the date of the prediction
    latest_date = df_sum.index[-1]
    # todo: map index to date

    predictions.apply(lambda x: x)

    # Calculate the root mean squared error
    rmse = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)

    # Create a new DataFrame that includes the original data and the predictions
    df_total = pd.concat([df_sum, predictions])
    # Create a heatmap using Plotly Express
    fig = px.line(df_total, labels={"index": "Date", "value": "Total Families"}, title="Total Families Over Time")
    fig.show()




if __name__ == '__main__':
    main()