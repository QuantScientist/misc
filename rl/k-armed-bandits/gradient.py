from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# noinspection PyUnresolvedReferences
from six.moves import range

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import seaborn as sns

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--k", dest="k", default=10, type=int)
parser.add_argument("--num_iteration", dest="num_iteration", default=1000, type=int)
parser.add_argument("--lr", dest="lr", default=0.1, type=float)
parser.add_argument("--seed", dest="seed", default=0, type=int)
args = parser.parse_args()

np.random.seed(args.seed)


def create_bandit(k, num_iteration):
    q_a = np.random.normal(loc=4., scale=1., size=k)
    R = np.zeros((k, num_iteration))
    for arm in range(k):
        R[arm, :] = np.random.normal(q_a[arm], scale=1.0, size=num_iteration)
    return q_a, R


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def one_hot(n, i):
    return (np.arange(n) == i).astype(np.float32)


def run(args, q_a, R):
    choices = np.zeros((args.k, args.num_iteration))
    H_a = np.zeros((args.k, args.num_iteration + 1))  # preference
    rewards = np.zeros((args.num_iteration,))
    baseline = np.zeros((args.num_iteration,))
    optimal_action_ratio = np.zeros((args.num_iteration,))

    for t in range(args.num_iteration):
        arm = np.argmax(softmax(H_a[:, t]))
        choices[arm, t] = 1
        rewards[t] = R[arm, t]
        is_optimal_action = (np.argmax(q_a) == arm).astype(np.float32)
        optimal_action_ratio[t] = (
            optimal_action_ratio[t-1] +
            1/(t+1) * (is_optimal_action - optimal_action_ratio[t-1]))
        baseline[t] = baseline[t-1] + 1/(t+1) * (rewards[t] - baseline[t-1])
        delta = (rewards[t] - baseline[t]) * (one_hot(args.k, arm) - softmax(H_a[:, t]))
        H_a[:, t+1] = H_a[:, t] + args.lr * delta

    return choices, H_a, rewards, optimal_action_ratio


averaged_ratio = np.zeros((args.num_iteration,))

for i in range(2000):
    q_a, R = create_bandit(args.k, args.num_iteration)
    choices, H_a, rewards, optimal_action_ratio = run(args, q_a, R)
    averaged_ratio += optimal_action_ratio

averaged_ratio /= 2000

plt.figure()
plt.plot(np.arange(args.num_iteration), averaged_ratio)
plt.savefig("optimal-action-ratio.png")
