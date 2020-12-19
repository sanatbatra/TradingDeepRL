## Deep Reinforcement Learning Swing Trading Bot - Soft Actor Critic

I worked on this project for the Deep Reinforcement Learning course at NYU Courant, Fall 2020.

### Introduction

Swing Trading is a trading strategy in financial markets where an asset is bought and held for a few days to several weeks, in an effort to profit from price changes or 'swings'.
Deep Reinforcement Learning is an ideal solution to the environment of swing trading, since the environment can be framed as a Markov's Decision Process (MDP).
As shown in Figure 1, the agent is to take an action (Buy, Hold or Sell) given the state it is in (current stock prices, holdings, external factors, etc.). At each step, it's action results in a reward (the profit it makes) and the agent needs to learn how to maximize this reward, by choosing the right set of actions. 

![RL problem diagram](https://spinningup.openai.com/en/latest/_images/rl_diagram_transparent_bg.png) 


Figure 1. Agent-environment interaction loop

### End Goal

I've been trading for the past 2 years and have been working on research in Deep Learning/AI for about the same time. It's time I put two and two together. I hope to build a trading bot that can consistently outperform the Dow Jones index by learning how to take advantage of the price fluctuations of each of its stocks.

### Assumptions

- I've assumed that there's no transaction fee to be paid on any buy/sell action. This is not a realistic assumption, but as we will see going forward, the usual transaction fee applied will only have a small impact on the final results.
- Another assumption is that the actions of the trading agent have no impact on the stock market, ie. the agent's actions do not affect the price movement of any of the stocks. This can be a realistic assumption if the agent is trading in small amounts that cannot really affect the demand and supply of the highly liquid stocks it's trading. For now, we can "afford" to ignore this issue.
- I've also made the assumptions that a stock can be bought at it's closing price. In reality, a stock cannot be purchased at or after the market closing time for the day. But, we could easily run the trading bot and send out orders 2-3 minutes before the market closes. A stock's price shouldn't vary much in the last few minutes, and this assumptions would only end up making a slight difference to our final results.

### Data

For this project, I've used the end of day prices and volume data (Open, High, Low, Close, Volume) of all 30 stocks in the Dow Jones Industrial Average index ranging from January 2009 - August 2020.

### Trading Environment Setup

Here's how my RL environment is set up:-

1. Observation Space: This is the information provided to the model to tell it what state it is in.
   - The current balance available to the agent to trade with. (1 dimension)
   - The quantity held of each of the 30 stocks. (30 dimensions)
   - The RSI, MACD, ADX and CCI of each of the 30 stocks. (30*4 = 120 dimensions)
   - The current closing price of each of the 30 stocks. (30 dimensions)
   
   Therefore, the observation space has 181 dimensions. 
   
   RSI, MACD, ADX and CCI are technical indicators that quantify the trend, momentum, etc. of the recent price fluctuations of a stock.
   
   Before including technical indicators in my state space, I experimented with encoding the closing prices from the past 'n' timesteps using LSTM, GRU and even 1-dimensional convolutional layers. I experimented with a varying values of 'n' and found that using an encoding of the history of prices in such a manner did not work well, i.e. I could not get my agent to learn using this information. Instead I used technical indicators, calculated using the historical price and volume data, to obtain information about the price history indirectly.  
 
 2. Action Space: The agent needs to take an action at each step.
    - 30 dimensions (1 action value for each stock), ranging from +1 which equates to buying the maximum quantity possible, to -1 which equates to selling the             maximum quantity possible. Maximum quantity is set to 10, therefore the agent cannot buy or sell more than 10 of any stock at one step.
 
 3. Reward: After taking an action, the agent gets a reward from the environment. This is the objective that it would try to maximize.
    - At each step, the reward is the increase in net worth of the portfolio from the previous step.
    
### Learning Algorithm

The trading agent needs to learn to maximize its objective (the net worth of the portfolio it is managing). For this purpose, I've used the Soft Actor Critic Algorithm (https://arxiv.org/abs/1801.01290). Soft Actor Critic (SAC) attempts to maximize the entropy of the agent alongwith maximizing its expected reward. Entropy is a measure of the randomness in the agents actions, and a learning algorithm that maximizes entropy helps greatly with exploration. Exploration is vital to training an agent in this trading environment because of its large action space (30 dimensions). Without exploration the agent would often get stuck in local optima of performing the same actions - only buying, or only selling, or only buying/selling a single stock. Hence, Soft Actor Critic is the ideal algorithm for this trading environment. Here, the optimal policy to be learnt is - 

![SAC policy equation](https://spinningup.openai.com/en/latest/_images/math/b86bf499707114c8789946df649871c5b9185b9d.svg)

where R is the reward function and H is the entropy function. ![alpha](https://spinningup.openai.com/en/latest/_images/math/900375490edee0019a5c54a311bf91de801a1642.svg) is the entropy coefficient (also called temperature) that decides how much importance to give to the entropy component in the above equation.  I've used an automatic gradient-based temperature tuning method that adjusts the expected entropy over the visited states to match a target value. I've also used prioritized replay with a smaller than usual replay size of 40,000.

### Experimental Setup

Each episode consisted of 252 steps (the average number of trading days in a year)

- Training Period: Training was done on data from January 2009 - December 2016
- Validation Period: Validation and tuning of hyperparameters was done on data from January 2017 - December 2017
- Testing Period: I only tested once after training and hyperparameter tuning, to ensure that there was no bias included in the test results in the form of tuning on the test set. Testing was done on data from January 2018 - August 2020.

To make sense of my test results, I compared the profit percentage made by my trained agent to that made by a baseline strategy of buying the Dow Jones index at the beginning of the test period and holding it till the end. This serves as a great baseline since most investors use a buy and hold strategy and assuming they picked a few stocks at random from the Dow Jones index, they can expect the same return percentage as the baseline strategy of buying and holding the index. If the agent has learnt how to succesfully buy low and sell high, it should get a better return percentage than the baseline strategy.


### Training Curve


![Training Curve](https://raw.githubusercontent.com/sanatbatra/TradingDeepRL/main/SoftActorCritic/plots/TrainScoreVsEpisodeSAC.png)


Figure 2. Agent Score (Profit) vs Episode number during training



As you can see from the training curve, the agent fits the training data quite well. It seems to overfit the relatively small training set since I've trained it for a large number of episodes (3500). Towards the end of training it gets ~50% return per episode (per year) on average, which is slightly unrealstic. To know if it can generalize to data from out of the training data, we should take a look at the validation curve.

### Validation Curve


![Validation Curve](https://raw.githubusercontent.com/sanatbatra/TradingDeepRL/main/SoftActorCritic/plots/EvalScoreVsEpisodeSAC.png)


Figure 3. Agent Score (Profit) vs Episode number on validation data



The agent almost consistently gets a profit that is much higher than what the baseline strategy could get on the validation data. That's a good sign that the agent is learning a profitable strategy that can generalize.

### Test Results
![Test Returns](https://raw.githubusercontent.com/sanatbatra/TradingDeepRL/main/SoftActorCritic/20180101-20200801_ReturnsComparison.png)


Figure 4. Agent and baseline returns during test period



The above plot is that of the cumulative returns made by both the trained agent and the baseline strategy throughout the testing period (January 2018 - August 2020). The agent outperforms the baseline by a large margin by the end of the testing period by getting a final cumulative return of 42.98% compared to a return of only 7.71% made by the baseline strategy.

### Conclusion and Future Work

The training and validation curves, as well as the test results point towards the fact that the agent is learning a profitable strategy that buys low and sells high. Furthermore, minimal hyperparameter tuning was done during the training process and that's always a good sign. Altough the agent is able to learn from the technical indicators provided to it, external information such as sentiment from news, could serve as a great addition to the observation space. Maybe the agent could even avoid the losses suffered during the market fall as a result of the coronavirus in March 2020, if provided with sentiment from news articles or blogs.
I will also be exploring the use of a history of technical indicators, instead of only the most recent values, encoded using LSTM/GRU/1-D conv. layers, as part of the state space. I believe this will provide the agent with even richer information, since professional traders usually use fluctuations of indicator values, the crossing over of two different indicators, etc. as signals to enter/exit trades.

