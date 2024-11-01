# LSTM-RainfallRunoff-Model
This repository contains a Python script written with PyTorch to train and test a LSTM-based regional rainfall-runoff hydrological.
- The data folder includes sample data for six drainage basins in CSV format. Each CSV file contains 29 input features and one target variable (runoff).
- The `lstm_hydro_model.py` script is self-sufficient. It loads the necessary modules, performs data preprocessing, trains the LSTM model for multiple basins during the training period, saves the model predictions in the output folder, and evaluates the model's performance using the Nash-Sutcliffe efficiency during the testing period.
