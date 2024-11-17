"""Global config file"""
from datetime import datetime
import os
import numpy as np

fldr = r"D:\My Drive\bin\latentfactors"
data_fldr = fr"{fldr}\data"

trn_start_dt = datetime(1963, 7, 31)  # Training start
bt_start_dt = datetime(2002, 12, 31)  # Backtest start
bt_end_dt = datetime(2023, 12, 31)  # Backtest end

factor_set = ["mktrf", "smb", "hml", "rmw", "cma"]  # Factors to study

n_states = 4  # Number of hidden states to use
estimation_freq = "W-Wed"
rebalance_freq = "M"
data_freq = "W-Wed"
obs_thresh = 260  # Five years of observations before estimating the RWLS
forecast_horizon = 4  # In months
