# plot.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Thisisjustadummyfunctiontogeneratesomearbitrarydata
def getdata():
    basecond=[[18,20,19,18,13,4,1],
        [20,17,12,9,3,0,0],
        [20,20,20,12,5,3,0]]
    cond1=[[18,19,18,19,20,15,14],
        [19,20,18,16,20,15,9],
        [19,20,20,20,17,10,0],
        [20,20,20,20,7,9,1]]
    cond2=[[20,20,20,20,19,17,4],
        [20,20,20,20,20,19,7],
        [19,20,20,19,19,15,2]]
    cond3=[[20,20,20,20,19,17,12],
        [18,20,19,18,13,4,1],
        [20,19,18,17,13,2,0],
        [19,18,20,20,15,6,0]]

    return basecond, cond1, cond2, cond3

# Helper for extracting scores from a string in the output log
def extractScore(strIn):
    strIn = strIn.strip()
    #print(strIn)

    fields = strIn.split(" ")
    if (strIn.startswith("EVAL EPISODE SCORE:")):
        fields = fields[1:]

    # EPISODE SCORE: 0.0 STEPS: 0 EPISODES: 0
    score = float(fields[2])
    steps = float(fields[4])

    if (fields[6].endswith("DONE")):
        fields[6] = fields[6][:-4]
        print("Updated: " + fields[6])

    episodes = float(fields[6])    

    # Score clipping
    if (score < 0):
        score = 0

    return {'score':score, 'steps': steps, 'episodes': episodes}

def getData(filename):
    trainScores = []
    evalScores = []

    # Open file
    with open(filename) as f:
        lines = f.readlines()

    # Extract data
    for line in lines:
        if (line.startswith("EPISODE SCORE:") and ("STEPS:" in line)):
            trainScores.append( extractScore(line) )
        if (line.startswith("EVAL EPISODE SCORE:") and ("STEPS:" in line)):
            evalScores.append( extractScore(line) )

    # Sort
    trainScores = sorted(trainScores, key = lambda x: x['steps'])
    evalScores = sorted(evalScores, key = lambda x: x['steps'])
    
    # Return
    return trainScores, evalScores


# Collapse (average) duplicates with the same step #
def filterData(dataIn):
    dataOut = []

    lastStepIdx = -1

    scores = []    
    episodes = []
    numSamples = 0

    for elem in dataIn:
        if (elem['steps'] != lastStepIdx):
            # Add old
            if (numSamples > 0):
                out = {'score':sum(scores)/numSamples, 'steps': lastStepIdx, 'episodes': sum(episodes)/numSamples}
                dataOut.append(out)                

            scores = []
            episodes = []

            scores.append(elem['score'])            
            episodes.append(elem['episodes'])
            lastStepIdx = elem['steps']
            numSamples = 1
            
        else:
            # Same
            scores.append(elem['score'])            
            episodes.append(elem['episodes'])
            numSamples += 1
            
        
    return dataOut


def rollingMeanHelper(data, startIdx, windowSize):
    if (len(data) > (startIdx + windowSize)):
        window = data[startIdx:startIdx+windowSize]
        avg = sum(window) / windowSize
        return avg
    
    return 0    ## TODO


def rollingMean(dataIn):
    windowSize = 20

    scores = [x['score'] for x in dataIn]
    steps = [x['steps'] for x in dataIn]
    episodes = [x['episodes'] for x in dataIn]
    idxs = []
    
    scoresRolling = []
    for i in range(0, len(scores)-windowSize):
        scoresRolling.append( rollingMeanHelper(scores, i, windowSize) )
        idxs.append(i)

    stepsRolling = []
    for i in range(0, len(scores)-windowSize):
        stepsRolling.append( rollingMeanHelper(steps, i, windowSize) )

    episodesRolling = []
    for i in range(0, len(scores)-windowSize):
        episodesRolling.append( rollingMeanHelper(episodes, i, windowSize) )


    return stepsRolling, scoresRolling, episodesRolling, idxs

#Load the data.
results = getdata()
fig = plt.figure()

#We will plot iterations 0...6
#xdata=np.array([0,1,2,3,4,5,6])/5.

#trainScores, evalScores = getData("out-findcrash10.txt")
trainScores, evalScores = getData("out-findcrash11simpl.txt")

trainScores = filterData(trainScores)
evalScores = filterData(evalScores)


ydataTrain = [x['score'] for x in trainScores]
ydataEval = [x['score'] for x in evalScores]

xdataTrain = [x['steps'] for x in trainScores]
xdataEval = [x['steps'] for x in evalScores]

print(xdataEval)
import collections
print([item for item, count in collections.Counter(xdataEval).items() if count > 1])


#Plot each line
#(may want to automate this part e.g. with a loop).

#sns.tsplot(time=[xdataTrain], data=[ydataTrain], color='r', linestyle='-')
#sns.tsplot(time=[xdataEval], data=[ydataEval], color='g', linestyle='--')


steps1, scores1, eps1, idxs1 = rollingMean(trainScores)
print(eps1)

steps2, scores2, eps2, idxs2 = rollingMean(evalScores)
print(eps2)

#sns.tsplot(time=[x], data=[y], color='r', linestyle='-')
sns.tsplot(time=[eps1], data=[scores1], color='r', linestyle='-')
sns.tsplot(time=[eps2], data=[scores2], color='g', linestyle='--')



#sns.tsplot(time=xdata, data=results[0], color='r', linestyle='-')
#sns.tsplot(time=xdata, data=results[1], color='g', linestyle='--')
#sns.tsplot(time=xdata, data=results[2], color='b', linestyle=':')
#sns.tsplot(time=xdata, data=results[3], color='k', linestyle='-.')

#Oury−axisis”successrate”here.
plt.ylabel("Score", fontsize=25)

#Ourx−axisisiterationnumber.
plt.xlabel("Episodes", fontsize=25, labelpad = -4 )

#Ourtaskiscalled”AwesomeRobotPerformance”
plt.title("Task 13", fontsize=30)

#Legend.
plt.legend(loc='bottomleft')

#Showtheplotonthescreen.
plt.show()
