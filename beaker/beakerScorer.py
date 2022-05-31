# beakerScorer.py

import os
import json
from statistics import mean


def loadFile(filenameIn):
    print(" * Loading " + filenameIn)
    f = open(filenameIn)
    data = json.load(f)
    return data


def sortByKey(dataIn):
    keys = dataIn.keys()

    dataOut = {}

    for key in keys:
        fields = key.split("-")        
        
        if (len(fields) == 2):
            # Eval key, of form 'episodeNum-sampleNum'
            newKey = int(fields[0] + fields[1].zfill(4))
            dataOut[newKey] = dataIn[key]
        else:
            # Regular key, of form 'episodeNum'
            dataOut[key] = dataIn[key]
        
    sortedKeys = sorted(dataOut.keys())

    sortedOut = []
    for key in sortedKeys:
        #print(key)
        sortedOut.append(dataOut[key])

    return sortedOut


def getLastNPercent(dataIn, proportion=0.10):
    length = len(dataIn)
    num = int(length * proportion)

    return dataIn[-num:]

def getAverageScore(dataIn):
    sum = 0
    numSamples = 0

    for elem in dataIn:
        history = elem['history']['history']
        lastStep = history[-1]
        score = float(lastStep['score'])

        # Clip score to 0-1
        if (score < 0):
            score = 0

        sum += score
        numSamples += 1

    avg = sum / numSamples    
    return avg


def doGetAverage(filenameIn):
    data = loadFile(filenameIn)
    sortedData = sortByKey(data)
    lastNPercent = getLastNPercent(sortedData, proportion=0.10)
    #return getAverageScore(sortedData)
    return getAverageScore(lastNPercent)



#
#   File I/O
#

# from https://thispointer.com/python-how-to-get-list-of-files-in-directory-and-sub-directories/
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles


#
#   Main
#

resultsPath = "/home/peter/github/tdqn-scienceworld/drrn/results/"
allFilenames = getListOfFiles(resultsPath)

errors = []

resultsByTask = {}
for taskNum in range(0, 30):
    print("Loading results for Task " + str(taskNum))
    resultsBySeed = []
    for seedNum in range(0, 5):
        substring = "seed" + str(seedNum) + "-task" + str(taskNum) + "-eval"
        filenames = [x for x in allFilenames if substring in x]
        print(filenames)
        if (len(filenames) > 1):
            print("ERROR: Multiple results for " + substring)
            errors.append(substring)
        elif (len(filenames) == 1):
            avg = doGetAverage(filenames[0])
            resultsBySeed.append(avg)

    print("results: " + str(resultsBySeed))

    avg = -1
    if (len(resultsBySeed) > 0):
        avg = mean(resultsBySeed)

    packed = {'samples': resultsBySeed, 'avg': avg}
    resultsByTask[taskNum] = packed


# Summary
print("")
print("-------------")
print("   SUMMARY ")
print("-------------")

print("Task#\tAvg\tNumSamples\tRawSamples")
for taskNum in range(0, 30):
    samples = resultsByTask[taskNum]['samples']
    avg = resultsByTask[taskNum]['avg']
    numSamples = len(samples)
    print(str(taskNum) + "\t" + str(avg) + " \t" + str(numSamples) + "\t" + "\t".join([str(x) for x in samples]))

print("-------------")
print("Errors: " + str(errors))

    