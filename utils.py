import datetime,random,os

import pandas as pd
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler,MinMaxScaler

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_data(path='daily_min_temperatures.csv'):
    '''
        path in [
            'daily_min_temperatures.csv','weather.csv','exchange_rate.csv',
            'ETTh1.csv','traffic.csv','electricity.csv'
        ]
    '''
    if (path == 'daily_min_temperatures.csv') and (not os.path.exists(f'data/{path}')):
        # https://github.com/jbrownlee/Datasets
        download_data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv')
        download_data['Date'] = pd.to_datetime(download_data['Date'])
        download_data = pd.DataFrame(pd.date_range(download_data['Date'].min(), download_data['Date'].max()), columns=['Date']).merge(data, on='Date', how='left')
        download_data.to_csv('data/daily_min_temperatures.csv',index=False)


    df=pd.read_csv(f'data/{path}').drop_duplicates() # weather数据集中有重复的，其他的没看，干脆都去重就好了
    if path == 'daily_min_temperatures.csv':
        df['Date'] = pd.to_datetime(df['Date'])
        df['Temp'] = df['Temp'].interpolate() # 用线性插值填充2个缺失值
        data = df.copy()
        valid_start_date='1990-11-01'
        valid_end_date='1990-11-30'
        train=data[data['Date'].between(data['Date'].min(),pd.to_datetime(valid_start_date) - datetime.timedelta(days=1))].copy()
        valid=data[data['Date'].between(pd.to_datetime(valid_start_date),pd.to_datetime(valid_end_date))].copy()
        test=data[data['Date'].between(pd.to_datetime(valid_end_date) + datetime.timedelta(days=1),data['Date'].max())].copy()

        data=data.set_index('Date')['Temp']
        train=train.set_index('Date')['Temp']
        valid=valid.set_index('Date')['Temp']
        test=test.set_index('Date')['Temp']
        return data,train,valid,test

    df['date']=pd.to_datetime(df['date'])
    # weather
    if path == 'weather.csv':
        data=pd.DataFrame(pd.date_range('2020-01-02 00:00:00','2020-12-31 23:50:00',freq='10min'),columns=['date'])
        data=data.merge(df[['date','OT']],on='date',how='left')
        data.loc[data['OT']==-9999,'OT']=np.nan # 修改异常值
        data['OT']=data['OT'].interpolate() # 用线性插值填充9个缺失值和上面的异常值
        valid_start_date='2020-12-30 00:00:00'
        valid_end_date='2020-12-30 23:50:00'
        train=data[data['date'].between(data['date'].min(),pd.to_datetime(valid_start_date) - datetime.timedelta(minutes=10))].copy()
        valid=data[data['date'].between(pd.to_datetime(valid_start_date),pd.to_datetime(valid_end_date))].copy()
        test=data[data['date'].between(pd.to_datetime(valid_end_date) + datetime.timedelta(minutes=10),data['date'].max())].copy()

    # exchange_rate
    if path == 'exchange_rate.csv':
        data=pd.DataFrame(pd.date_range('1990-01-01','2010-09-30',freq='d'),columns=['date'])
        data=data.merge(df[['date','OT']],on='date',how='left')
        valid_start_date='2010-08-01'
        valid_end_date='2010-08-31'
        train=data[data['date'].between(data['date'].min(),pd.to_datetime(valid_start_date) - datetime.timedelta(days=1))].copy()
        valid=data[data['date'].between(pd.to_datetime(valid_start_date),pd.to_datetime(valid_end_date))].copy()
        test=data[data['date'].between(pd.to_datetime(valid_end_date) + datetime.timedelta(days=1),data['date'].max())].copy()

    # ETTh1
    if path == 'ETTh1.csv':
        data=pd.DataFrame(pd.date_range('2016-07-01 00:00:00','2018-06-25 23:00:00',freq='h'),columns=['date'])
        data=data.merge(df[['date','OT']],on='date',how='left')
        valid_start_date='2018-06-14 00:00:00'
        valid_end_date='2018-06-19 23:00:00'
        train=data[data['date'].between(data['date'].min(),pd.to_datetime(valid_start_date) - datetime.timedelta(hours=1))].copy()
        valid=data[data['date'].between(pd.to_datetime(valid_start_date),pd.to_datetime(valid_end_date))].copy()
        test=data[data['date'].between(pd.to_datetime(valid_end_date) + datetime.timedelta(hours=1),data['date'].max())].copy()

    # traffic
    if path == 'traffic.csv':
        data=pd.DataFrame(pd.date_range('2016-07-02 00:00:00','2018-07-01 23:00:00',freq='h'),columns=['date'])
        data=data.merge(df[['date','OT']],on='date',how='left')
        valid_start_date='2018-06-20 00:00:00'
        valid_end_date='2018-06-25 23:00:00'
        train=data[data['date'].between(data['date'].min(),pd.to_datetime(valid_start_date) - datetime.timedelta(hours=1))].copy()
        valid=data[data['date'].between(pd.to_datetime(valid_start_date),pd.to_datetime(valid_end_date))].copy()
        test=data[data['date'].between(pd.to_datetime(valid_end_date) + datetime.timedelta(hours=1),data['date'].max())].copy()

    # electricity
    if path == 'electricity.csv':
        data=pd.DataFrame(pd.date_range('2016-07-02 00:00:00','2019-07-01 23:00:00',freq='h'),columns=['date'])
        data=data.merge(df[['date','OT']],on='date',how='left')
        valid_start_date='2019-06-20 00:00:00'
        valid_end_date='2019-06-25 23:00:00'
        train=data[data['date'].between(data['date'].min(),pd.to_datetime(valid_start_date) - datetime.timedelta(hours=1))].copy()
        valid=data[data['date'].between(pd.to_datetime(valid_start_date),pd.to_datetime(valid_end_date))].copy()
        test=data[data['date'].between(pd.to_datetime(valid_end_date) + datetime.timedelta(hours=1),data['date'].max())].copy()

    data=data.set_index('date')['OT']
    train=train.set_index('date')['OT']
    valid=valid.set_index('date')['OT']
    test=test.set_index('date')['OT']

    return data,train,valid,test

class TimeSeriesDataSet(Dataset):

    def __init__(self, data, seq_len, valid_len, test_len, scaler='',
                is_valid=False, is_test=False, is_all=False):
        self.data_raw = data
        self.scaler = scaler
        if scaler != '':
            assert scaler in ['minmax','std']
            if scaler == 'minmax':
                self.scaler = MinMaxScaler().fit(data[:-valid_len-test_len].values.reshape(-1, 1))
            elif scaler == 'std':
                self.scaler = StandardScaler().fit(data[:-valid_len-test_len].values.reshape(-1, 1))

            self.data = self.scaler.transform(data.values.reshape(-1,1))
        else:
            self.data = data.values.reshape(-1,1)

        self.seq_len = seq_len
        self.valid_len = valid_len
        self.test_len = test_len

        self.is_valid = is_valid
        self.is_test = is_test
        self.is_all = is_all

        self.sequences_data = self.create_sequences_data()

    def __len__(self):
        return len(self.sequences_data)

    def create_sequences_data(self):
        if self.is_valid:
            idx_start = len(self.data) - self.valid_len - self.test_len - self.seq_len
            idx_end = len(self.data) - self.seq_len - self.test_len
            
        elif self.is_test:
            idx_start = len(self.data) - self.test_len - self.seq_len
            idx_end = len(self.data) - self.seq_len
            
        elif self.is_all:
            idx_start = 0
            idx_end = len(self.data) - self.seq_len

        else:
            idx_start = 0
            idx_end = len(self.data) - self.seq_len - self.valid_len - self.test_len

        sequences_data = []
        for idx in range(idx_start,idx_end):
            start = idx
            end = start+self.seq_len
            seq = self.data[start:end]
            label = self.data[end]
            sequences_data.append([seq,label])

        return sequences_data

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.sequences_data[idx][0]).float()
        label = torch.from_numpy(self.sequences_data[idx][1]).float()
        return seq, label
    
    def inverse_transform(self, data):
        if self.scaler != '':
            return self.scaler.inverse_transform(data)
        else:
            return data

class CoronaVirusPredictor(nn.Module):

    def __init__(self, args):
        super(CoronaVirusPredictor, self).__init__()

        self.n_hidden = args.n_hidden
        self.seq_len = args.seq_len
        self.n_layers = args.n_layers
        self.device = args.device

        self.lstm = nn.LSTM(
            input_size=args.n_features,
            hidden_size=args.n_hidden,
            num_layers=args.n_layers,
            dropout=args.dropout
        )

        self.linear = nn.Linear(in_features=args.n_hidden, out_features=1)

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len,
                        self.n_hidden).to(self.device),
            torch.zeros(self.n_layers, self.seq_len,
                        self.n_hidden).to(self.device)
        )

    def forward(self, sequences):
        lstm_out, self.hidden = self.lstm(sequences.view(len(sequences), self.seq_len, -1), self.hidden)
        last_time_step = lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred