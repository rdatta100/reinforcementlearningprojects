# Made by Rani Datta
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import random
import statistics
import math 

# Will be implementing the Thompson Sampling Algorithm in this class
# This class specifically takes two arms that both have true normal distributions 
class ThompsonSamplingTwoNormals:
    def __init__(self, trueArms) -> None:
        self.trueArms = trueArms # the true distributions of the arms
        self.observedArms = [[0,1], [0,1]] # what we think the distributions are based on observed rewards
        self.data = [[],[]] # keeps track of the reward observed for both arms
        self.n = 0 # the number of trials done, starting at 0

    def model1(self, j):
        return np.random.normal(self.trueArms[j][0], trueArms[j][1])

    def model2(self, its):
        for i in range(its): # for loop for the amount of trials passed in: its
            max = float('-inf') 
            pickedArm = -1

            for j in range(2): # loops through both arms
                randVal = self.model1(j) # samples each arm using the true distribution
                if randVal > max: # if this value is greater than the max
                    max = randVal # updates the max value
                    pickedArm = j # picks the arm since it outputted a higher value

            self.data[pickedArm].append(max) # adds the value observed from the picked arm to its respective data array
            sampleMu = statistics.mean(self.data[pickedArm]) # calculates the mean of the data

            if len(self.data[pickedArm]) <= 1: # calculates the sample variance of the data
                sampleVar = 0 #  if there is only one data point, the sample variance is 0
            else:
                sampleVar = statistics.variance(self.data[pickedArm]) 

            priorMu = self.observedArms[pickedArm][0] # retrieves the prior mean which is stored in the observedArms array
            priorVar = self.observedArms[pickedArm][1]**2 # retrieves the prior variance (squared since the observedArms array stores std)

            self.n += 1 # increments the number of trials done by 1

            postMu = self.calcPostMu(self.n, sampleMu, sampleVar, priorMu, priorVar) # calculates posterior mean using helper function
            postVar = self.calcPostVar(self.n, sampleVar, priorVar) # calculates posterior variance using helper function

            self.observedArms[pickedArm][0] = postMu # updates the observedArms array using the posterior mean found
            self.observedArms[pickedArm][1] = math.sqrt(postVar) # updates the observedArms array using the posterior variance found

    
    def calcPostMu(self, n, sampleMu, sampleVar, priorMu, priorVar): # helper function to calculate posterior mean
        # documentation of this equation can be found in equations folder
        num = sampleVar*priorMu + (self.n*priorVar*sampleMu)
        den = n*priorVar + sampleVar
        if den == 0.0:
            return 0
        return num / den
    
    def calcPostVar(self, n, sampleVar, priorVar): # helper function to calculate posterior variance
        # documentation of this equation can be found in equations folder
        if sampleVar == 0:
            return 1.0
        num = sampleVar*priorVar
        den = n*priorVar + sampleVar
        return num / den

    def getBestArm(self): # retrieves the best arm based on which has the higher observed mean
        if self.observedArms[0][0] > self.observedArms[0][0]:
            return 'Arm 1 is the best!'
        else:
            return 'Arm 2 is the best!'

trueArms = [[0,1],[3,5]] # [[arm1 true mean, arm1 true std], [arm2 true mean, arm2 true std]]
TSobj = ThompsonSamplingTwoNormals(trueArms) # creates ThompsonSamplingTwoNormals object using the trueArms array of the two arms' normal distributions
TSobj.model2(1000) # run for 1000 trials
print(TSobj.getBestArm()) # prints the best observed arm