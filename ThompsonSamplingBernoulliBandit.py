import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import random

def plot():
  fig, axs = plt.subplots(2, 2)
  axs[0, 0].plot(x, stats.beta.pdf(x, alphas[0], betas[0]))
  axs[0, 0].set_title('Machine 1')
  axs[0, 1].plot(x, stats.beta.pdf(x, alphas[1], betas[1]), 'tab:orange')
  axs[0, 1].set_title('Machine 2')
  axs[1, 0].plot(x, stats.beta.pdf(x, alphas[2], betas[2]), 'tab:green')
  axs[1, 0].set_title('Machine 3')
  axs[1, 1].plot(x, stats.beta.pdf(x, alphas[3], betas[3]), 'tab:red')
  axs[1, 1].set_title('Machine 4')


  for ax in axs.flat:
      ax.label_outer()

machines_prob = [0.1, 0.2, 0.7, 0.8]

alphas = [1, 1, 1, 1]
betas = [1, 1, 1, 1]
plot()

for i in range(50):
  max = -1
  index = -1
  for j in range(4):
    rand_val = random.betavariate(alphas[j], betas[j])
    if (rand_val > max):
      max = rand_val
      index = j

  play = np.random.binomial(1, machines_prob[index], 1)
  if (play[0] == 1):
    alphas[index] += 1
  else:
    betas[index] += 1

  plot()


