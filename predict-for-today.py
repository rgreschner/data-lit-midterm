#!/usr/bin/env python

# Import libraries for date handling.
import delorean
import time
from datetime import datetime

# Import libraries for general data handling.
from alpha_vantage.timeseries import TimeSeries
import numpy as np
import pandas as pd

# Import scikit-learn model fit and evaluation.
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Import Seaborn for plotting.
import seaborn as sns

# Stock symbol to do prediction for.
STOCK_SYMBOL = 'MDB'
# Use live data from AlphaVantage API.
USE_LIVE_DATA = False
# AlphaVantage API Key.
ALPHAVANTAGE_API_KEY = '!!! your API key goes here !!!'
# Ratio for train / test split.
SPLIT_POINT = 0.7
# Random seed for train / test data selection.
SPLIT_SEED = 42
# Path to CSV file containing historical stock data.
HISTORICAL_CSV_DATA_PATH = './mdb-historical-data.csv'


def load_daily_data_from_alpha_vantage():
    """
    Load daily data from AlphaVantage API.
    """
    print('Loading daily data from AlphaVantage API...')
    ts = TimeSeries(key=ALPHAVANTAGE_API_KEY, output_format='pandas')
    loaded_data_df, _ = ts.get_daily(STOCK_SYMBOL)
    print('Loaded daily data from AlphaVantage API.')
    return loaded_data_df

def load_daily_data_from_csv():
    """
    Load daily data from previously exported CSV.
    """
    loaded_data_df = pd.read_csv(HISTORICAL_CSV_DATA_PATH)         .set_index('date')
    print('Loaded daily data from historical CSV.')
    return loaded_data_df


def prepare_daily_data(data):
    """
    Prepare input data for prediction and visual inspection etc.
    """
    # Select only relevant columns in source.
    stock_data_only_close = data[['4. close']]         .reset_index()         .reset_index()         .rename({'4. close': 'close'}, axis=1)
    # Map date to date structure from DeLorean lib for easier manipulation and parsing.
    stock_data_only_close['date_delorean'] = stock_data_only_close['date'].apply(lambda date: delorean.parse(date, dayfirst=False))
    # Format date human-readable for input data validation, e.g. to prevent flipped day/month values.
    # Use ISO 8601 for uniform display.
    stock_data_only_close['date'] = stock_data_only_close['date_delorean'].apply(lambda date: date.date)
    stock_data_only_close['date_formatted'] = stock_data_only_close['date'].apply(lambda date: date.isoformat())
    stock_data_only_close['date_unix'] = stock_data_only_close['date'].apply(lambda date: time.mktime(date.timetuple()))
    # Drop unnecessary columns.
    stock_data_only_close = stock_data_only_close.drop(['date_delorean'], axis=1)
    return stock_data_only_close

# Load data from configured source.
loaded_data_df = load_daily_data_from_alpha_vantage() if USE_LIVE_DATA else load_daily_data_from_csv()
# Prepare input data.
prepared_data_df = prepare_daily_data(loaded_data_df)

prepared_data_df.tail(10)

def convert_data_frame_to_vectors(prepared_data_df):
    """
    Convert input data frame to vector input
    for scikit-learn regressor.
    """
    # Uses Unix timestamp as independent variable
    # for predicition input.
    dates = prepared_data_df['date_unix'].tolist()
    prices = prepared_data_df['close'].tolist()
    dates = np.reshape(dates, (len(dates), 1))
    prices = np.reshape(prices, (len(prices), 1))
    return dates, prices

# Prepare vector inputs for model fit.
dates, prices = convert_data_frame_to_vectors(prepared_data_df)

def split_data(dates, prices):
    """
    Split input data at configured split point
    in order to generate train / test data sets.
    """
    return train_test_split(dates, prices, test_size=1-SPLIT_POINT, random_state=SPLIT_SEED)

def get_split_info_df(x_train, x_test):
    """
    Get information for train / test data set split
    for debugging / accountability reasons.
    """
    train_test_split_info_df = pd.DataFrame([
        {'Name': 'Train Set', 'Count': len(x_train)},
        {'Name': 'Test Set', 'Count': len(x_test)}
    ])
    train_test_split_info_df = train_test_split_info_df[['Name', 'Count']]
    return train_test_split_info_df

x_train, x_test, y_train, y_test = split_data(dates, prices)

train_test_split_info_df = get_split_info_df(x_train, x_test)
train_test_split_info_df

regressor = LinearRegression()
regressor.fit(x_train, y_train)

def predict_for_dates(prediction_dates):
    """
    Predict closing stock price for
    supplied date values.
    """
    # Actually predict values.
    predicted_closes = regressor.predict(prediction_dates)
    # Map supplied date values to DataFrame for easier post-processing.
    dates_df = pd.DataFrame(prediction_dates, columns=['date_unix']).reset_index()
    # Output prediction result in column 'predicted_close' in DataFrame
    # including date as well.
    predicted_closes_df = pd.DataFrame(predicted_closes, columns=['predicted_close'])         .reset_index()         .merge(dates_df)
    return predicted_closes_df

# Predict stock price for today/now.
date_now = datetime.now()
unix_time_now = time.mktime(date_now.timetuple())
prediction_dates = [[unix_time_now]]
predicted_closes_df = predict_for_dates(prediction_dates)

# Output result.
print('Predicted close for ' + str(date_now) + ': ' + str(predicted_closes_df['predicted_close'].iloc[0]))