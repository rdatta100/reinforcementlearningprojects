# Made by Rani Datta
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import random

# Will be implementing the Epsilon Greedy Algorithm in this class
class eGreedy:
  def __init__(self, numArms, rewardFunction, eps=0.1 ):
    self.numArms = numArms # number of arms
    self.armsAvg = np.zeros(numArms) # the average reward of the arms, starting off at 0
    self.timesPicked = np.ones(numArms) # the number of times that arm is picked, starts off at 1
    self.rewardFunction = rewardFunction # the reward function of each of the arm's distribution
    self.numIterations = 0  # number of trials to run
    self.eps = eps # declare a epsilon value, default is 0.1
    self.rewards = [] # keeps track of the reward observed
  
  def model1(self, arm):
    return self.rewardFunction[arm]()
  
  def model2(self, its):
    self.numIterations += its # increment the number of iterations by its: the # of trials 

    for i in range(its): # run for however many trials specified
      randNum = np.random.uniform(0,1,1)[0] # generate random number between 0 and 1

      if randNum < self.eps: # if that number is less than epsilon
        arm = np.random.choice(self.numArms, 1)[0] # choose random arm
      else: 
        arm = self.getBestArm() # otherwise do greedy approach by choosing the best arm
        
      reward = self.model1(arm) # observe reward of the arm chosen
      num = (self.armsAvg[arm] * self.timesPicked[arm] + reward) # calculate the total reward for that arm and add the reward observed
      denom = (self.timesPicked[arm] + 1.0) # calculate the total amount of times picked
      self.armsAvg[arm] = num / denom # calculate the updated average reward
      self.timesPicked[arm] += 1 # increment the amount of times this arm was picked by 1
      self.rewards.append(reward) # append the reward received

  def getBestArm(self): # the best arm is the arm with the highest average reward observed
    return np.argmax(self.armsAvg) 
    
  def plotter(self): # plots the rewards
    rews = np.cumsum(self.rewards).astype(float)
      
    for i in range(len(rews)):
      rews[i] = rews[i] / (i + 1.0)
    plt.plot(range(1, len(rews) + 1), rews)

  def getArmAvg(self): # returns the average reward array for all arms
    return self.armsAvg


functions = [ # reward functions for each of the arms aka their distributions
    lambda : np.random.randn() + 5, 
    lambda : np.random.randint(0, 10), 
]

Eobj = eGreedy(2, functions, 0.1) # create eGreedy object with 2 arms, the distributions above, and an episolon value of 0.1
Eobj.model2(1000) # run 1000 trials 
bestArm = Eobj.getBestArm() # retrive the best arm observed
if bestArm == 0:
  print('Arm 1 is the best!')
else:
  print('Arm 2 is the best!')
