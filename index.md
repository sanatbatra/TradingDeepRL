## Deep Reinforcement Learning Swing Trading Bot - Soft Actor Critic

I worked on this project for the Deep Reinforcement Learning course at NYU Courant, Fall 2020.

### Introduction

Swing Trading is a trading strategy in financial markets where an asset is bought and held for a few days to several weeks, in an effort to profit from price changes or 'swings'.
Deep Reinforcement Learning is an ideal solution to the environment of swing trading, since the environment can be framed as a Markov's Decision Process (MDP).
The agent is to take an action (Buy, Hold or Sell) given the state it is in (current stock prices, holdings, external factors, etc.). At each step, it's action results in a reward (the profit it makes) and the agent needs to learn how to maximize this reward, by choosing the right set of actions. 

### End Goal

I've been trading for the past 2 years and have been working on research in Deep Learning/AI for about the same time. It's time I put two and two together. I hope to build a trading bot that can consistently outperform the Dow Jones index by learning how to take advantage of the price fluctuations of each of its stocks.

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
 
 2. Action Space: The agent needs to take an action at each step.
    - 30 dimensions (1 action value for each stock), ranging from +1 which equates to buying the maximum quantity possible, to -1 which equates to selling the             maximum quantity possible. Maximum quantity is set to 10, therefore the agent cannot buy or sell more than 10 of any stock at one step.
 
 3. Reward: After taking an action, the agent gets a reward from the environment. This is the objective that it would try to maximize.
    - Increase in net worth of the portfolio from the previous step.
    
### Learning Algorithm

The trading agent needs to learn to maximize its objective (the net worth of the portfolio it is managing). For this purpose, I've used the Soft Actor Critic Algorithm (https://arxiv.org/abs/1801.01290). Soft Actor Critic (SAC) attempts to maximize the entropy of the agent alongwith maximizing its expected reward. Entropy is a measure of the randomness in the agents actions, and a learning algorithm that maximizes entropy helps greatly with exploration. Exploration is vital to training an agent in this trading environment because of its large action space (30 dimensions). Without exploration the agent would often get stuck in local optima of performing the same actions - only buying, or only selling, or only buying/selling a single stock. Hence, Soft Actor Critic is the ideal algorithm for this trading environment. 

### Experimental Setup

Each episode consisted of 252 steps (the average number of trading days in a year)

- Training Period: Training was done on data from January 2009 - December 2016
- Validation Period: Validation and tuning of hyperparameters was done on data from January 2017 - December 2017
- Testing Period: I only tested once after training and hyperparameter tuning, to ensure that there was no bias included in the test results in the form of tuning on the test set. Testing was done on data from January 2018 - August 2020.

To make sense of my test results, I compared the profit percentage made by my trained agent to that made by a baseline strategy of buying the Dow Jones index at the beginning of the test period and holding it till the end. This serves as a great baseline since most investors use a buy and hold strategy and assuming they picked a few stocks at random from the Dow Jones index, they can expect the same return percentage as the baseline strategy of buying and holding the index. If the agent has learnt how to succesfully buy low and sell high, it should get a better return percentage than the baseline strategy.


### Training Curve

![Training Curve](https://github.com/sanatbatra/TradingDeepRL/blob/main/SoftActorCritic/plots/TrainScoreVsEpisodeSAC.png)

As you can see from the training curve, the agent fits the training data quite well. It seems to overfit the relatively small training set since I've trained it for a large number of episodes (3500). Towards the end of training it gets ~50% return per episode (per year) on average, which is slightly unrealstic. To know if it can generalize to data from out of the training data, we should take a look at the validation curve.

### Validation Curve

![Validation Curve](https://github.com/sanatbatra/TradingDeepRL/blob/main/SoftActorCritic/plots/EvalScoreVsEpisodeSAC.png)

The agent almost consistently, gets a profit that is much higher than what the baseline strategy could get on the validation data. That's a good sign that the agent is learning a profitable strategy that can generalize.

### Test Results

![Test Returns](https://github.com/sanatbatra/TradingDeepRL/blob/main/SoftActorCritic/20180101-20200801_ReturnsComparison.png)





```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/sanatbatra/TradingDeepRL/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
