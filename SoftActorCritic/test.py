import torch
from collections import defaultdict
from SAC.environment import *
import pandas as pd

from SAC.model import GaussianPolicy
from SAC.sac import grad_false

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_results(price_map, quantity_map, returns_list_bot, returns_list_djia, djia_df, start_date, end_date):

    save_name = '%s-%s' % (start_date, end_date)

    plt.clf()
    plt.figure(figsize=(11, 7))
    plt.plot(djia_df['Date'].iloc[:len(returns_list_bot)], returns_list_bot)
    plt.plot(djia_df['Date'].iloc[:len(returns_list_bot)], returns_list_djia)
    plt.xticks(rotation=45)
    plt.legend(['Agent Returns', 'Dow Jones Buy and Hold Returns (Baseline)'], loc='upper left')
    plt.ylabel('Cumulative Returns (%)')
    plt.xlabel('Date')
    plt.savefig('%s_ReturnsComparison.png' % save_name)

    for stock in price_map:
        price_list = price_map[stock]
        quantity_list = quantity_map[stock]
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(2)

        ax1.plot(list(range(len(price_list))), price_list)
        ax2.plot(list(range(len(price_list))), quantity_list)

        ax1.set(xlabel='Step Number', ylabel='Price')
        ax2.set(xlabel='Step Number', ylabel='Quantity Held')
        for ax in fig.get_axes():
            ax.label_outer()

        plt.savefig('./test_results/%s_%s_TestStockPrice&QuantityHeld.png' % (save_name,stock))


def calculate_sharpe_ratio(l):
    df = pd.DataFrame(l)
    df.columns = ['net_worth']
    df['daily_return_percentage'] = df.pct_change(1)
    return (252 ** 0.5) * df['daily_return_percentage'].mean() / df['daily_return_percentage'].std()


def run():
    df = pd.read_csv('./done_data.csv', index_col=0)
    start_date = 20180101
    end_date = 20200801
    val_df = df[(df.datadate >= start_date) & (df.datadate < end_date)]
    val_df = val_df.sort_values(['datadate', 'tic'], ignore_index=True)
    val_df.index = val_df.datadate.factorize()[0]

    djia_df = pd.read_csv('./data/DowJones.csv')
    djia_df['Date'] = pd.to_datetime(djia_df['Date'])
    mask = ('2018-01-01' <= djia_df['Date']) & (djia_df['Date'] < '2020-08-01')
    djia_df = djia_df.loc[mask]

    no_obs = 5
    max_episode_steps = (len(val_df)//30)-1
    verbose = False

    test_env = TradingEnv(val_df, no_obs, 30, max_steps=max_episode_steps, verbose=verbose, test_mode=True)
    device = torch.device("cpu")

    policy = GaussianPolicy(
        test_env.observation_space.shape[0],
        test_env.action_space.shape[0],
        hidden_units=[400, 300]).to(device)

    checkp = torch.load('./saved_models/policy.pth', map_location='cpu')
    policy.load_state_dict(checkp)
    grad_false(policy)

    def exploit(state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            _, _, action = policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    day = 0
    state = test_env.reset()
    episode_reward = 0.
    done = False

    price_map = defaultdict(list)
    quantity_map = defaultdict(list)
    returns_list_bot = [0.0]
    returns_list_djia = [0.0]
    date_list = [djia_df.iloc[0].Date]
    net_worth_agent_list = [test_env.net_worth]
    net_worth_buyhold_list = [djia_df.iloc[0].Close]

    while not done:

        price_list = val_df.loc[day, :].adjcp.values.tolist()
        tic_list = val_df.loc[day, :].tic.values.tolist()
        action = exploit(state)
        next_state, reward, done, _ = test_env.step(action)

        for stock_idx in range(len(action)):
            price_map[tic_list[stock_idx]].append(price_list[stock_idx])
            quantity_map[tic_list[stock_idx]].append(test_env.quantity_held[stock_idx])

        day += 1

        returns_list_bot.append((test_env.net_worth-test_env.starting_balance)*100.0/test_env.starting_balance)
        returns_list_djia.append((djia_df.iloc[day].Close - djia_df.iloc[0].Close)*100.0/djia_df.iloc[0].Close)
        date_list.append(djia_df.iloc[day].Date)
        net_worth_agent_list.append(test_env.net_worth)
        net_worth_buyhold_list.append(djia_df.iloc[day].Close)

        episode_reward += reward
        state = next_state

    print('Profit: %s,   Baseline: %s' % ((test_env.net_worth - test_env.starting_balance), test_env.buy_hold_baseline))

    sharpe_agent = calculate_sharpe_ratio(net_worth_agent_list)
    sharpe_buyhold = calculate_sharpe_ratio(net_worth_buyhold_list)
    print('Agent Sharpe Ratio: %s,   Buy Hold Sharpe Ratio: %s' %(sharpe_agent, sharpe_buyhold))

    plot_results(price_map, quantity_map, returns_list_bot, returns_list_djia, djia_df, start_date, end_date)


if __name__ == '__main__':
    run()