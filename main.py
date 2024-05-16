import datetime
import json
import plotly.express as px
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt


def get_participation(xls):
    participation = {}
    for sheet_name, df in xls.items():
        for index, row in df.iterrows():
            if index < 2:
                continue

            first_name = row['Vorname/Kind']
            last_name = row['Name']
            full_name = f"{first_name} {last_name}"
            for col in df.columns:
                if type(col) is datetime.datetime:
                    key = f"{col.year}-{str(col.month).zfill(2)}-{str(col.day).zfill(2)}"

                    if row[col] == 1.0:
                        encoded_value = 1
                    else:
                        encoded_value = 0

                    if key in participation:
                        participation[key][full_name] = encoded_value
                    else:
                        participation[key] = {full_name: encoded_value}

    return participation


def prepare_data(participation):
    df = pd.DataFrame(participation).T

    # Fill missing participants that later joined with "0"
    df = df.fillna(0)

    # Drop the families where each date is "0"
    df = df.loc[:, ~(df == 0).all(axis=0)]

    # drop all dates where all families are 0
    df = df.loc[~(df == 0).all(axis=1)]

    # Sort the DataFrame by its index (the dates)
    df = df.sort_index()

    # Sum up the participation of families for each date
    df = df.sum(axis=1)

    # Convert the index to date
    df.index = pd.to_datetime(df.index).to_period('W')

    return df

def main():
    # read Excel file
    xls = pd.read_excel('data/Eltern-Kind Turnen Freitag 16-17.xlsx', sheet_name=None)
    # xls is a dictionary where the keys are the sheet names and the values are the dataframes
    participation = get_participation(xls)

    # Prepare the data
    df = prepare_data(participation)

    # save min index for plotting
    min_index = min(df.index)

    # Split the data into training and test sets
    train = df
    test = df[int(0.8 * len(df)):]

    # Fit an ARIMA model to the training data
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()

    # Use the fitted model to make predictions on the test data
    predictions = model_fit.predict(start=min(test.index), end=max(test.index), dynamic=False, freq='W')

    # plot the Training Data
    df = df.to_frame()
    df['data_type'] = 'Training Data'
    df.rename(columns={0: 'value'}, inplace=True)

    # plot the Predictions on Test Data
    test_df = predictions.to_frame()
    test_df['data_type'] = 'Test Data'
    test_df.rename(columns={'predicted_mean': 'value'}, inplace=True)

    # plot the forecast
    forecast = model_fit.get_forecast(steps=8)
    predicted_values = forecast.predicted_mean
    forecast_df = pd.DataFrame(predicted_values)
    forecast_df['data_type'] = 'Forecast'
    forecast_df.rename(columns={'predicted_mean': 'value'}, inplace=True)
    confidence_intervals = forecast.conf_int()


    # Reset the index of the predictions DataFrame
    # test_df = test_df.reset_index(drop=True)
    # Concatenate the DataFrames
    # Convert the PeriodIndex of df to a RangeIndex
    # df.reset_index(drop=True, inplace=True)

    # Concatenate the DataFrames
    df_total = pd.concat([df, test_df, forecast_df], ignore_index=False)

    # Convert the PeriodIndex to a DateTimeIndex
    df_total.index = pd.date_range(start=min_index.end_time, periods=len(df_total))

    # sort index
    df_total = df_total.sort_index()

    # Use Plotly Express to create the line plot
    fig = px.line(df_total, x=df_total.index, y='value', color='data_type', color_discrete_sequence=['blue', 'red', 'green'])

    # Show the plot
    fig.show()

if __name__ == '__main__':
    main()
