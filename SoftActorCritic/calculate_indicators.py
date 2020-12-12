import pandas as pd
from ta import add_all_ta_features
from ta.utils import dropna

stock_names = ['AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS',
               'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO',
               'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V',
               'VZ', 'WBA', 'WMT']

full_df = None
for stock in stock_names:
    df = pd.read_csv('./data/%s.csv' % stock)
    df['date'] = df['Date'].replace('-', '')
    df['stock'] = stock
    processed_df = add_all_ta_features(df, open='Open', high='High', low='Low', close='Close', volume='Volume')

    if full_df is None:
        full_df = df
    else:
        df = df.reset_index(drop=True)
        full_df = pd.concat([full_df, df], axis=0)

full_df = full_df.sort_values(['date','stock']).reset_index(drop=True)
full_df.to_csv('data_indicators.csv')