# Score histogram

from collections import defaultdict
import math

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
        #print("Updated: " + fields[6])

    episodes = float(fields[6])    

    # Score clipping
    if (score < 0):
        score = 0

    return {'score':score, 'steps': steps, 'episodes': episodes}

def getData(filename, isBeakerLog=False):
    trainScores = []
    evalScores = []

    # Open file
    with open(filename) as f:
        lines = f.readlines()

    # Extract data
    for line in lines:

        # If it's a Beaker log, strip out everything before the first space
        #print("BEFORE: " + line)
        if (isBeakerLog):
            line = line.split(' ', 1)[1]
        #print(" AFTER: " + line)

        if (line.startswith("EPISODE SCORE:") and ("STEPS:" in line)):
            trainScores.append( extractScore(line) )
        if (line.startswith("EVAL EPISODE SCORE:") and ("STEPS:" in line)):
            evalScores.append( extractScore(line) )

    # Sort
    trainScores = sorted(trainScores, key = lambda x: x['steps'])
    evalScores = sorted(evalScores, key = lambda x: x['steps'])
    
    # Return
    return trainScores, evalScores


# Histogram
def mkHistogram(dataIn):
    count = defaultdict(lambda:0)
    maxEpisode = 0
    
    print("Score\tCount\tProportion")

    totalCount = 0    
    for elem in dataIn:        
        count[ int(elem['score']) ] += 1
        totalCount += 1

        if (elem['episodes'] > maxEpisode):
            maxEpisode = elem['episodes']

    for key in sorted(count.keys()):
        rawCount = count[key]
        proportion = "{:.3f}".format((count[key] / totalCount))
        print(str(key) + "\t" + str(rawCount) + "\t" + str(proportion))

    print("(Maximum episode: " + str(maxEpisode) + ")")

    #print(count)


def getLastNPercent(dataIn, proportion):
    startIdx = math.floor((1 - proportion) * len(dataIn))
    return dataIn[startIdx:]

#
#   Main
#

#trainScores, evalScores = getData("out-findcrash12.txt")
#trainScores, evalScores = getData("out-findcrash15-task18.txt")
#trainScores, evalScores = getData("out-findcrash14.txt")
#trainScores, evalScores = getData("../results-beaker/example.log", isBeakerLog=True)

trainScores, evalScores = getData("../results-beaker/sciworld-drrn-8x100k-task6-seed0.log", isBeakerLog=True)

#print(trainScores)

print("Training (seen):")
mkHistogram(trainScores)

print("")

print("Evaluation (unseen):")
mkHistogram(evalScores)

print("")
print("--------")
print("")

lastProportion = 0.10
print("Training (seen, last " + str(lastProportion) + "):")
mkHistogram( getLastNPercent(trainScores, lastProportion) )

print("")
print("Evaluation (unseen, last " + str(lastProportion) + "):")
mkHistogram( getLastNPercent(evalScores, lastProportion) )


