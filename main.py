import datetime

import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objs import Figure
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults


def get_participation(xls: dict[str, pd.DataFrame]) -> dict[str, dict[str, int]]:
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


def prepare_data(participation: dict[str, dict[str, int]]) -> pd.Series:
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


def plot(train_df: pd.DataFrame,
         test_df: pd.DataFrame,
         forecast_df: pd.DataFrame,
         confidence_df: pd.DataFrame) -> Figure:
    # create a new figure
    fig = go.Figure()

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
        showlegend=True,
        name='upper'
    ))
    fig.add_trace(go.Scatter(
        x=confidence_df.index,
        y=confidence_df['lower y'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(0,0,0,0.2)',
        showlegend=True,
        name='lower'
    ))

    return fig


def split(df: pd.Series) -> tuple[pd.Series, pd.Series]:
    # Split the data into training and test sets
    train = df[:int(0.8 * len(df))]
    test = df[int(0.8 * len(df)):]

    return train, test


def train_model(train: pd.Series, test: pd.Series) -> tuple[pd.Series, ARIMAResults]:
    # Fit an ARIMA model to the training data
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()

    # Use the fitted model to make predictions on the test data
    predictions = model_fit.predict(start=min(test.index), end=max(test.index), dynamic=False)

    return predictions, model_fit


def build_train_for_plot(train: pd.Series) -> pd.DataFrame:
    train_df = train.to_frame()
    train_df['data_type'] = 'Train Data'
    train_df.rename(columns={0: 'value'}, inplace=True)
    train_df.index = pd.date_range(start=min(train_df.index).end_time, end=max(train_df.index).end_time,
                                   periods=len(train_df))
    return train_df


def build_test_for_plot(test: pd.Series, start: pd.Timestamp) -> pd.DataFrame:
    test_df = test.to_frame()
    test_df['data_type'] = 'Test Data'
    test_df.rename(columns={'predicted_mean': 'value'}, inplace=True)
    test_df.index = pd.date_range(start=start, periods=len(test_df), freq='W')
    return test_df


def build_forecast_for_plot(
        model_fit: ARIMAResults,
        len_test: int,
        start: pd.Timestamp) -> tuple[
    pd.DataFrame, pd.DataFrame]:
    # Forecast the next 8 weeks and the test data
    forecast = model_fit.get_forecast(steps=8 + len_test)
    predicted_values = forecast.predicted_mean
    forecast_df = pd.DataFrame(predicted_values)
    forecast_df['data_type'] = 'Forecast'
    forecast_df.rename(columns={'predicted_mean': 'value'}, inplace=True)
    forecast_df.index = pd.date_range(start=start, periods=len(forecast_df), freq='W')

    confidence_intervals = forecast.conf_int()
    confidence_df = pd.DataFrame(confidence_intervals)
    confidence_df['data_type'] = 'Confidence Interval'
    confidence_df.rename(columns={0: 'lower', 1: 'upper'}, inplace=True)
    confidence_df.index = pd.date_range(start=start, periods=len(confidence_df), freq='W')
    return forecast_df, confidence_df


def main():
    # read Excel file
    xls = pd.read_excel('data/Eltern-Kind Turnen Freitag 16-17.xlsx', sheet_name=None)
    # xls is a dictionary where the keys are the sheet names and the values are the dataframes
    participation = get_participation(xls)

    # Prepare the data
    df = prepare_data(participation)

    # Split the data into training and test sets
    train_series, test_series = split(df)
    predictions, model_fit = train_model(train_series, test_series)

    train_df = build_train_for_plot(train_series)
    test_df = build_test_for_plot(predictions, max(train_df.index))
    forecast_df, confidence_df = build_forecast_for_plot(model_fit, len(test_series), max(train_series.index).end_time)
    fig = plot(train_df, test_df, forecast_df, confidence_df)
    fig.show()


if __name__ == '__main__':
    main()
