# Stock Data Prediction Model using Linear Regression

## Introduction

This is a midterm coursework done for the Data Lit course of the School of AI.

The goal of the midterm was to generate stock closing price prediction using some data retrieval, cleanup and model training to generate a prediction for a freely choosen stock symbol. For the later, I chose MongoDB, Inc. (NASDAQ: MDB).

For a Jupyter Notebook detailling the solution, see `data-lit-midterm.ipynb`. In order to use this, you need to manually upload the CSV file `mdb-historical-data.csv` containing historical data into Google Colab. Also due to some installed libraries a runtime restart is required.

For a script version which predicts the stock price for today, run `predict-for-today.py`.

The CSV file `mdb-historical-data.csv` contains historical data for MongoDB Inc. Stocks ranging from 2018/10/12 to 2019/03/08.