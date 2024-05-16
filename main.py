import datetime
import json
import plotly.express as px
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import plotly.graph_objects as go

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
    df.sort_index(inplace=True)

    # Split the data into training and test sets
    train = df[:int(0.8 * len(df))]
    test = df[int(0.8 * len(df)):]

    # Fit an ARIMA model to the training data
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()

    # Use the fitted model to make predictions on the test data
    predictions = model_fit.predict(start=min(test.index), end=max(test.index), dynamic=False, freq='W')

    # plot the Training Data
    train_df = train.to_frame()
    train_df['data_type'] = 'Train Data'
    train_df.rename(columns={0: 'value'}, inplace=True)
    train_df.index = pd.date_range(start=min(train_df.index).end_time,  end=max(train_df.index).end_time, periods=len(train_df))
    train_df.sort_index(inplace=True)

    # plot the Predictions on Test Data
    test_df = predictions.to_frame()
    test_df['data_type'] = 'Test Data'
    test_df.rename(columns={'predicted_mean': 'value'}, inplace=True)
    test_df.index = pd.date_range(start=max(train_df.index), periods=len(test_df), freq='W')
    test_df.sort_index(inplace=True)

    # plot the forecast
    forecast = model_fit.get_forecast(steps=8 + len(test))
    predicted_values = forecast.predicted_mean
    forecast_df = pd.DataFrame(predicted_values)
    forecast_df['data_type'] = 'Forecast'
    forecast_df.rename(columns={'predicted_mean': 'value'}, inplace=True)
    forecast_df.index = pd.date_range(start=max(train.index).end_time, periods=len(forecast_df), freq='W')
    forecast_df.sort_index(inplace=True)
    confidence_intervals = forecast.conf_int()


    # Create a new DataFrame for the confidence intervals
    confidence_df = pd.DataFrame(confidence_intervals)
    confidence_df['data_type'] = 'Confidence Interval'
    confidence_df.rename(columns={0: 'lower', 1: 'upper'}, inplace=True)
    # Convert the PeriodIndex to a DateTimeIndex
    confidence_df.index = pd.date_range(start=max(train.index).end_time, periods=len(confidence_df), freq='W')
    confidence_df.sort_index(inplace=True)

    # Concatenate the DataFrames
    df_total = pd.concat([train_df, test_df, forecast_df, confidence_df], ignore_index=True)

    # Sort index
    df_total = df_total.sort_index()

    # create new figure
    fig = go.Figure()

    # Use Plotly Express to create the line plot
    # fig = px.line(df, x=df.index, y='value', color='red')

    # Add the training data to the plot
    fig.add_trace(go.Scatter(
        x=train_df.index,
        y=train_df['value'],
        mode='lines',
        showlegend=True,
        name='Training Data'
    ))

    # Add the test data to the plot
    fig.add_trace(go.Scatter(
        x=test_df.index,
        y=test_df['value'],
        mode='lines',
        showlegend=True,
        name='Test Data'
    ))

    # Add the forecast to the plot
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df['value'],
        mode='lines',
        showlegend=True,
        name='Forecast'
    ))

    # Add the confidence intervals to the plot
    fig.add_trace(go.Scatter(
        x=confidence_df.index,
        y=confidence_df['upper y'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        name='upper'
    ))
    fig.add_trace(go.Scatter(
        x=confidence_df.index,
        y=confidence_df['lower y'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(0,0,0,0.2)',
        showlegend=False,
        name='lower'
    ))

    # Show the plot
    fig.show()

if __name__ == '__main__':
    main()
