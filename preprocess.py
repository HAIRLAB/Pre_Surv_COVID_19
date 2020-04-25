# -- coding:utf-8 --
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import xgboost as xgb

import utils


def preprocess_time_series():
    # 读入375病人的数据，并指定第一列PATIENT_ID为第一索引，第二索引为RE_DATE（检查时间）
    data = pd.read_excel('data/time_series_375_prerpocess.xlsx', index_col=[0, 1])
    data = data.dropna(thresh=6)
    data.to_parquet('data/time_series_375.parquet')
    # 读入110病人的数据，并指定第一列PATIENT_ID为第一索引，第二索引为RE_DATE（检查时间）
    data = pd.read_excel('data/time_series_test_110_preprocess.xlsx', index_col=[0, 1])
    data.to_parquet('data/time_series_test_110.parquet')


if __name__ == '__main__':
    preprocess_time_series()
