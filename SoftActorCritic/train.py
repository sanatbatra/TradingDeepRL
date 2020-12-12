import os
from datetime import datetime
import pandas as pd
from environment import *
from sac import *


def run():
    df = pd.read_csv('./done_data.csv', index_col=0)
    train_df = df[(df.datadate >= 20090101) & (df.datadate < 20170101)]
    val_df = df[(df.datadate >= 20170101) & (df.datadate < 20180101)]
    train_df = train_df.sort_values(['datadate', 'tic'], ignore_index=True)
    train_df.index = train_df.datadate.factorize()[0]
    val_df = val_df.sort_values(['datadate', 'tic'], ignore_index=True)
    val_df.index = val_df.datadate.factorize()[0]

    no_obs = 5
    max_episode_steps = 250
    verbose = False

    env = TradingEnv(train_df, no_obs, 30, max_steps=max_episode_steps, verbose=verbose)
    val_env = TradingEnv(val_df, no_obs, 30, max_steps=max_episode_steps, verbose=verbose, test_mode=True)

    configs = {
        'num_episodes': 3500,
        'batch_size': 256,
        'lr': 0.00007,
        'hidden_units': [400, 256],
        'memory_size': 40000,
        'gamma': 0.99,
        'tau': 0.005,
        'entropy_tuning': True,
        'ent_coef': 0.2,
        'multi_step': 1,
        'per': True,  # prioritized experience replay
        'alpha': 0.6,
        'beta': 0.4,
        'beta_annealing': 0.0001,
        'grad_clip': None,
        'updates_per_step': 1,
        'start_steps': 10000,
        'log_interval': 10,
        'target_update_interval': 1,
        'eval_interval': 2500,
        'cuda': True,
        'max_episode_steps': max_episode_steps,
        'no_obs': no_obs
    }

    agent = SacAgent(env=env, val_env=val_env, **configs)
    agent.run()


run()
