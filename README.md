# LSTM-RainfallRunoff-Model
This repository contains a Python script written in PyTorch to train and test an LSTM-based rainfall-runoff hydrological model across multiple basins.
- The data folder includes sample data for six drainage basins in CSV format. Each CSV file contains 29 input features and one target variable (runoff).
- The `lstm_hydro_model.py` script is self-sufficient. It loads the necessary modules, performs data preprocessing, trains the LSTM model for multiple basins during the training period, and evaluates the model's performance using the Nash-Sutcliffe efficiency during the testing period.
