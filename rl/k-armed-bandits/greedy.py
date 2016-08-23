from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# noinspection PyUnresolvedReferences
from six.moves import range
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import seaborn as sns


parser = argparse.ArgumentParser()
parser.add_argument("--k", dest="k", type=int, default=10)
parser.add_argument("--num_iteration", dest="num_iteration", type=int, default=1000)
parser.add_argument("--epsilon", dest="epsilon", type=float, default=0.)
parser.add_argument("--seed", dest="seed", type=int, default=0)
parser.add_argument("--Q_init", type=float, default=0., dest="Q_init")
args = parser.parse_args()

np.random.seed(args.seed)

assert 0 <= args.epsilon <= 1

q_a = np.random.normal(loc=0.0, scale=1.0, size=args.k)
R = np.zeros((args.k, args.num_iteration))
for arm in range(args.k):
    R[arm, :] = np.random.normal(q_a[arm], scale=1.0, size=args.num_iteration)
optimal_action = np.argmax(q_a)


def run(args, epsilon=args.epsilon):
    choices = np.zeros((args.k, args.num_iteration))  # choices[i, j] = 1 if arm_i was selected at j-th iteration
    Q_a = np.zeros((args.k, args.num_iteration)) + args.Q_init
    rewards = np.zeros((args.num_iteration,))

    for t in range(args.num_iteration):
        if t == 0 or np.random.uniform(low=0., high=1.) < args.epsilon:
            arm = np.random.randint(low=0, high=args.k)  # high is not included
        else:
            arm = np.argmax(Q_a[:, t-1])

        total_reward_a = (rewards * choices[arm]).sum()
        count_a = choices[arm].sum()
        if count_a == 0:
            count_a = 1
        Q_a[arm, t:] = total_reward_a / count_a

        choices[arm, t] = 1
        rewards[t] = R[arm, t]

    return choices, Q_a, rewards


choices, Q_a, rewards = run(args, args.epsilon)

# plot
fig, axes = plt.subplots(figsize=(16, 16))

plt.subplot2grid((3, 2), (0, 0))
plt.title("epsilon=%.1f" % args.epsilon)
df = pd.DataFrame(data=R.transpose([1, 0]))
sns.violinplot(df)
sns.utils.axlabel("Arm", "Reward")

plt.subplot2grid((3, 2), (0, 1))
plt.title("epsilon=%.1f" % args.epsilon)
R_ = np.zeros((args.k, args.num_iteration))
for arm in range(args.k):
    R_[arm, :] = np.random.normal(Q_a[arm], scale=1.0, size=args.num_iteration)
df = pd.DataFrame(data=R_.transpose([1, 0]))
sns.violinplot(df)
sns.utils.axlabel("Arm", "Estimated reward")

plt.subplot2grid((3, 2), (1, 0))
plt.title("epsilon=%.1f" % args.epsilon)
sns.barplot(np.arange(args.k, step=1), choices.sum(axis=1))
sns.utils.axlabel("Arm", "Number of times taken")

p1 = plt.subplot2grid((3, 2), (1, 1), rowspan=2)
sns.utils.axlabel("Iteration", "Average reward")
p2 = plt.subplot2grid((3, 2), (2, 0))
sns.utils.axlabel("Iteration", "Optimal action")
epsilons = np.union1d(np.asarray([args.epsilon]), np.arange(0, 1.1, step=0.3))
for epsilon in epsilons:
    choices, _, rewards = run(args, epsilon)
    is_optimal = (optimal_action == np.argmax(choices, axis=0)).astype(np.float32)

    total_reward = 0.
    optimal_choice_count = 0.
    average_rewards = []
    optimal_choice_proportions = []
    for t in range(args.num_iteration):
        average_rewards += [(total_reward + rewards[t]) / (t + 1)]
        total_reward += rewards[t]

        optimal_choice_proportions += [(optimal_choice_count + is_optimal[t]) / (t + 1)]
        optimal_choice_count += is_optimal[t]

    p1.plot(np.arange(args.num_iteration, step=1), average_rewards, label=("epsilon=%.1f" % epsilon))
    p2.plot(np.arange(args.num_iteration, step=1), optimal_choice_proportions, label=("epsilon=%.1f" % epsilon))
p1.legend()
p2.legend()

plt.savefig("result-greedy.png")
