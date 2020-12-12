import gym
from gym import spaces
import numpy as np
import random
import statistics
import math

REWARD_SCALING = 1e-3


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, no_of_days, no_of_stocks, max_steps=250, verbose=False, test_mode=False):
        super(TradingEnv, self).__init__()
        self.test_mode = test_mode
        self.no_of_days = no_of_days
        self.verbose = verbose
        self.starting_balance = 100000.0
        self.current_balance = self.starting_balance
        self.net_worth = self.current_balance
        self.df = df
        self.no_of_stocks = no_of_stocks
        self.current_day = 0
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.no_of_stocks,))
        self.observation_space = spaces.Box(low=-1 * np.inf, high=np.inf, shape=((1 + (30 * 6)),),
                                            dtype=np.float32)
        self.quantity_held = [0 for _ in range(self.no_of_stocks)]
        self.episode_no = 0
        self.no_bought = 0
        self.no_sold = 0
        self.max_steps = max_steps
        self.steps_count = 0
        self.buy_hold_baseline = 0.0
        self.max_buy_sell_quantity = 10  # max 10 shares per trade

    def step(self, action):
        reward = self._take_action(action)
        self.steps_count += 1
        self.current_day += 1
        obs = self._next_observation()
        done = False
        if self.steps_count == self.max_steps:
            # if self.currently_holding == 1:
            #   reward = (self.df.iloc[self.current_day].Open - self.bought_at_price)* self.quantity_held

            #   self.net_worth = self.current_balance + (self.quantity_held * self.df.iloc[self.current_day].Open)
            done = True
            self.episode_no += 1
            if self.verbose:
                print('Episode No: ', self.episode_no)
                print('Episode Total Reward: ', self.net_worth - self.starting_balance)
                print('Profit Percentage: ', (self.net_worth - self.starting_balance) * 100.0 / self.starting_balance)
                print('Bought: ', self.no_bought)
                print('Sold: ', self.no_sold)
                print('Did nothing: ', self.no_do_nothing)
                print('Failed to buy: ', self.no_failed_buy)
                print('Failed to sell: ', self.no_failed_sell)
                print('\n\n')
        return obs, reward, done, {}

    def reset(self):

        self.quantity_held = [0 for _ in range(self.no_of_stocks)]
        self.current_balance = self.starting_balance
        self.net_worth = self.current_balance
        self.steps_count = 0
        self.buy_hold_baseline = 0.0

        if self.test_mode:
            self.current_day = 0
        else:
            self.current_day = random.randint(0, (len(self.df)//self.no_of_stocks) - 1 - self.max_steps)

        start_list = self.df.loc[self.current_day, :].adjcp.values.tolist()
        end_list = self.df.loc[self.current_day+self.max_steps, :].adjcp.values.tolist()
        buying_amount_each = (self.starting_balance / float(self.no_of_stocks))
        for stock_idx in range(self.no_of_stocks):
            max_quant = buying_amount_each/start_list[stock_idx]

            self.buy_hold_baseline += (max_quant * end_list[stock_idx])

        self.buy_hold_baseline -= self.starting_balance
        return self._next_observation()

    # def normalize_list(self, l):
    #     l_max = max(l)
    #     l_min = min(l)
    #     max_min_diff = l_max - l_min
    #     # diff_percentage = max_min_diff/l_min
    #     return [(ele - l_min) / max_min_diff for ele in l]
    #     # return [ele for ele in l], 0.0
    #     # m = statistics.mean(l)
    #     # std = statistics.pstdev(l)
    #     # return [(ele - m)/std for ele in l]

    # def difference(self, l):
    #     l1 = [(l[i + 1] - l[i]) / l[i] for i in range(len(l) - 1)]
    #     return l1

    def _next_observation(self):
        data = self.df.loc[self.current_day, :]
        self.current_prices = data.adjcp.values.tolist()

        obs = [self.current_balance] + \
               data.adjcp.values.tolist() + \
               self.quantity_held[:] + \
               data.macd.values.tolist() + \
               data.rsi.values.tolist() + \
               data.cci.values.tolist() + \
               data.adx.values.tolist()

        return obs

    def _take_action(self, action):

        # if self.test_mode:
        #   print(action)
        # act_sum = 0.0
        # for a in action:
        #     if a > 0.0:
        #         act_sum += a

        # current_balance_each = (self.current_balance/float(len(self.stock_names)))
        action = action * self.max_buy_sell_quantity
        for stock_idx in range(self.no_of_stocks):
            # Sell
            if action[stock_idx] < 0.0:
                # sell_quantity = round(self.quantity_held*abs(action))
                sell_quantity = min(self.quantity_held[stock_idx], round(abs(action[stock_idx])))
                # sell_quantity = round(self.quantity_held[stock] * abs(action[idx]))
                # reward = (self.df.iloc[self.current_day].Open - self.bought_at_price)*sell_quantity
                self.current_balance += (sell_quantity * self.current_prices[stock_idx])
                self.quantity_held[stock_idx] -= sell_quantity
                self.no_sold += sell_quantity
                #
                # self.currently_holding[stock] = 0
                # self.bought_at_price[stock] = 0.0
                # action_taken = 0

            # Buy
            elif action[stock_idx] > 0.0:
                # buy_quantity = min(self.current_balance//self.df.iloc[self.current_day].Open,
                #                    round((self.current_balance * action)/self.df.iloc[self.current_day].Open))
                buy_quantity = min(self.current_balance//self.current_prices[stock_idx], round(action[stock_idx]))
                # buy_quantity = (self.current_balance * action[idx] / act_sum) // self.df_map[stock].iloc[
                #     self.current_day].Open
                # self.currently_holding[stock] = 1
                # self.bought_at_price[stock] = self.df_map[stock].iloc[self.current_day].Open

                self.current_balance -= (buy_quantity * self.current_prices[stock_idx])
                self.quantity_held[stock_idx] += buy_quantity
                self.no_bought += buy_quantity
                # action_taken =

        net_worth = self.current_balance
        next_prices = self.df.loc[self.current_day+1, :].adjcp.values.tolist()

        for stock_idx in range(self.no_of_stocks):
            net_worth += (self.quantity_held[stock_idx] * next_prices[stock_idx])

        # reward = (net_worth - worst_net_worth) - (best_net_worth - net_worth)
        reward = (net_worth - self.net_worth) * REWARD_SCALING
        # reward = (net_worth - (prev_balance + (prev_quantity_held * self.df.iloc[self.current_day+1].Open)))
        # reward = math.log(net_worth/self.net_worth)

        self.net_worth = net_worth

        return reward

    def render(self, mode='human', close=False):
        profit = self.net_worth - self.starting_balance
        print(f'Day: {self.current_day}')
        print(f'Balance: {self.current_balance}')
        print(f'Shares held: {self.quantity_held}')
        print(f'Net worth: {self.net_worth}')
        print(f'Profit: {profit}')
        print('Bought: ', self.no_bought)
        print('Sold: ', self.no_sold)
        print('Did Nothing: ', self.no_do_nothing)
        print('Failed to buy: ', self.no_failed_buy)
        print('Failed to sell: ', self.no_failed_sell)
        print('\n')
