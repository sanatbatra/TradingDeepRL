import os
import numpy as np
import torch
from torch.optim import Adam
from rltorch.memory import MultiStepMemory, PrioritizedMemory
from collections import defaultdict
from model import *
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import random

# from pyvirtualdisplay import Display

# virtual_display = Display(visible=0, size=(1400, 900))
# virtual_display.start()
import torch


def to_batch(state, action, reward, next_state, done, device):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    action = torch.FloatTensor([action]).view(1, -1).to(device)
    reward = torch.FloatTensor([reward]).unsqueeze(0).to(device)
    next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
    done = torch.FloatTensor([done]).unsqueeze(0).to(device)
    return state, action, reward, next_state, done


def update_params(optim, network, loss, grad_clip=None, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    if grad_clip is not None:
        for p in network.modules():
            torch.nn.utils.clip_grad_norm(p.parameters(), grad_clip)
    optim.step()


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


def hard_update(target, source):
    target.load_state_dict(source.state_dict())


def grad_false(network):
    for param in network.parameters():
        param.requires_grad = False


class SacAgent:

    def __init__(self, env, val_env, num_episodes=5000, batch_size=256,
                 lr=0.0003, hidden_units=[256, 256], memory_size=1e6,
                 gamma=0.99, tau=0.005, entropy_tuning=True, ent_coef=0.2,
                 multi_step=1, per=False, alpha=0.6, beta=0.4,
                 beta_annealing=0.0001, grad_clip=None, updates_per_step=1,
                 start_steps=10000, log_interval=10, target_update_interval=1,
                 eval_interval=1000, cuda=True, max_episode_steps=251, no_obs=40):
        self.env = env
        self.val_env = val_env

        torch.backends.cudnn.deterministic = True  # 'True' harms performance.
        torch.backends.cudnn.benchmark = False

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")
        print(self.device)

        self.policy = GaussianPolicy(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            hidden_units=hidden_units).to(self.device)
        self.critic = TwinnedQNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            hidden_units=hidden_units).to(self.device)
        self.critic_target = TwinnedQNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            hidden_units=hidden_units).to(self.device).eval()

        # copy parameters of the learning network to the target network
        hard_update(self.critic_target, self.critic)
        # disable gradient calculations of the target network
        grad_false(self.critic_target)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr, eps=1e-7)
        self.q1_optim = Adam(self.critic.Q1.parameters(), lr=lr, eps=1e-7)
        self.q2_optim = Adam(self.critic.Q2.parameters(), lr=lr, eps=1e-7)

        if entropy_tuning:
            # Target entropy is -|A|.
            self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
            # We optimize log(alpha), instead of alpha.
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=lr)
        else:
            # fixed alpha
            self.alpha = torch.tensor(ent_coef).to(self.device)

        if per:
            # replay memory with prioritied experience replay
            # See https://github.com/ku2482/rltorch/blob/master/rltorch/memory
            self.memory = PrioritizedMemory(
                memory_size, self.env.observation_space.shape,
                self.env.action_space.shape, self.device, gamma, multi_step,
                alpha=alpha, beta=beta, beta_annealing=beta_annealing)
        else:
            # replay memory without prioritied experience replay
            # See https://github.com/ku2482/rltorch/blob/master/rltorch/memory
            self.memory = MultiStepMemory(
                memory_size, self.env.observation_space.shape,
                self.env.action_space.shape, self.device, gamma, multi_step)

        self.train_rewards = []

        self.max_episode_steps = max_episode_steps

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.num_episodes = num_episodes
        self.tau = tau
        self.per = per
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.gamma_n = gamma ** multi_step
        self.entropy_tuning = entropy_tuning
        self.grad_clip = grad_clip
        self.updates_per_step = updates_per_step
        self.log_interval = log_interval
        self.target_update_interval = target_update_interval
        self.eval_interval = eval_interval
        self.eval_episode_nos = []
        self.eval_scores = []

    def run(self):

        ep_nos_avg = []
        reward_avg = []
        for i in range(self.num_episodes):
            stop_training = self.train_episode()

            ep_nos = [x + 1 for x in range(len(self.train_rewards))]
            plt.clf()
            plt.plot(ep_nos, self.train_rewards)
            if i > 50:
                reward_avg.append(sum(self.train_rewards[-50:]) / len(self.train_rewards[-50:]))
                ep_nos_avg.append(i + 1)
                plt.plot(ep_nos_avg, reward_avg)
                plt.legend(['Score', 'Mean Score 50 Episodes'], loc='upper left')
            else:
                plt.legend(['Score'], loc='upper left')
            plt.ylabel('Train Score')
            plt.xlabel('Episode Number')
            plt.savefig('TrainScoreVsEpisode.png')

            if stop_training:
                break

        print('Training Done.')

    def is_update(self):
        return len(self.memory) > self.batch_size and self.steps >= self.start_steps

    def act(self, state):
        if self.start_steps > self.steps:
            action = self.env.action_space.sample()
        else:
            action = self.explore(state)
        return action

    def explore(self, state):
        # act with randomness
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    def exploit(self, state):
        # act without randomness
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, action = self.policy.sample(state)
            # action, _, _ = self.policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.critic(states, actions)
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_entropies, _ = self.policy.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) + self.alpha * next_entropies

        target_q = rewards + (1.0 - dones) * self.gamma_n * next_q

        return target_q

    def train_episode(self):
        self.episodes += 1
        episode_reward = 0.
        episode_steps = 0
        done = False
        state = self.env.reset()
        stop_training = False

        while True:
            action = self.act(state)
            next_state, reward, done, _ = self.env.step(action)

            self.steps += 1
            episode_steps += 1
            episode_reward += reward

            # ignore done if the agent reach time horizons
            # (set done=True only when the agent fails)
            if episode_steps >= self.max_episode_steps:
                masked_done = False
            else:
                masked_done = done

            if self.per:
                batch = to_batch(state, action, reward, next_state, masked_done, self.device)
                with torch.no_grad():
                    curr_q1, curr_q2 = self.calc_current_q(*batch)
                target_q = self.calc_target_q(*batch)
                error = torch.abs(curr_q1 - target_q).item()
                # We need to give true done signal with addition to masked done
                # signal to calculate multi-step rewards.
                self.memory.append(state, action, reward, next_state, masked_done, error, episode_done=done)
            else:
                # We need to give true done signal with addition to masked done
                # signal to calculate multi-step rewards.
                self.memory.append(state, action, reward, next_state, masked_done, episode_done=done)

            state = next_state

            # self.env.render()
            if self.is_update():
                for _ in range(self.updates_per_step):
                    self.learn()

            if self.steps % self.eval_interval == 0:
                eval_reward = self.evaluate()
                self.save_models()

            if done:
                break

        self.train_rewards.append(self.env.net_worth - self.env.starting_balance)

        print(f'episode: {self.episodes:<4}  '
              f'episode steps: {episode_steps:<4}  '
              f'profit: {self.env.net_worth - self.env.starting_balance:<5.1f}  '
              f'baseline: {self.env.buy_hold_baseline:<5.1f}')

        return stop_training

    def learn(self):
        self.learning_steps += 1
        if self.learning_steps % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        if self.per:
            # batch with indices and priority weights
            batch, indices, weights = \
                self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
            # set priority weights to 1 when we don't use PER.
            weights = 1.

        q1_loss, q2_loss, errors, mean_q1, mean_q2 = \
            self.calc_critic_loss(batch, weights)
        policy_loss, entropies = self.calc_policy_loss(batch, weights)

        update_params(
            self.q1_optim, self.critic.Q1, q1_loss, self.grad_clip, retain_graph=True)
        update_params(
            self.q2_optim, self.critic.Q2, q2_loss, self.grad_clip, retain_graph=True)
        update_params(
            self.policy_optim, self.policy, policy_loss, self.grad_clip, retain_graph=True)

        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(entropies, weights)
            update_params(self.alpha_optim, None, entropy_loss)
            self.alpha = self.log_alpha.exp()

        if self.per:
            # update priority weights
            self.memory.update_priority(indices, errors.cpu().numpy())

    def calc_critic_loss(self, batch, weights):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)
        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)
        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch

        # We re-sample actions to calculate expectations of Q.
        sampled_action, entropy, _ = self.policy.sample(states)
        # expectations of Q with clipped double Q technique
        q1, q2 = self.critic(states, sampled_action)
        q = torch.min(q1, q2)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = torch.mean((- q - self.alpha * entropy) * weights)
        return policy_loss, entropy

    def calc_entropy_loss(self, entropy, weights):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropy).detach()
            * weights)
        return entropy_loss

    def evaluate(self):
        self.eval_episode_nos.append(self.episodes)
        state = self.val_env.reset()
        episode_reward = 0.
        done = False

        while True:
            action = self.exploit(state)
            next_state, reward, done, _ = self.val_env.step(action)
            # self.env.render()
            episode_reward += reward
            state = next_state
            if done:
                break

        self.eval_scores.append(self.val_env.net_worth - self.val_env.starting_balance)
        total_reward = (self.val_env.net_worth - self.val_env.starting_balance)

        buy_hold_base_total = self.val_env.buy_hold_baseline
        plt.clf()
        plt.plot(self.eval_episode_nos, self.eval_scores)
        plt.plot(self.eval_episode_nos, [buy_hold_base_total for _ in range(len(self.eval_episode_nos))])
        plt.legend(['Score', 'Buy and Hold baseline'], loc='upper left')
        plt.ylabel('Eval Score')
        plt.xlabel('Episode Number')
        plt.savefig('EvalScoreVsEpisode.png')

        print('-' * 60)
        print(f'Num steps: {self.steps:<5}  '
              f'Evaluation reward: {total_reward:<5.1f}  '
              f'Buy Hold Baseline: {buy_hold_base_total:<5.1f}')
        print('-' * 60)
        return total_reward

    def save_models(self):
        self.policy.save('policy.pth')
        self.critic.save('critic.pth')
        self.critic_target.save('critic_target.pth')

