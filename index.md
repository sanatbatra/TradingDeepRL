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

The trading agent needs to learn to maximize its objective (the net worth of the portfolio it is managing). For this purpose, I've used the Soft Actor Critic Algorithm (https://arxiv.org/abs/1801.01290). 

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
