
from scienceworld_python_api import VirtualEnv
from multiprocessing import Process, Pipe

import numpy as np
import random
import time
import sys


def sanitizeInfo(infoIn):
    # Convert from py4j.java_collections.JavaList to python list
    recastList = []
    for elem in infoIn['valid']:
        recastList.append(elem)

    info = {'moves': infoIn['moves'],
            'score': infoIn['score'],
            'look': infoIn['look'],
            'inv': infoIn['inv'],
            #'valid': ['look', 'eat', 'inventory'],
            'valid': recastList,
            'taskDesc': infoIn['taskDesc']
        }

    return info


def sanitizeObservation(obsIn, infoIn):
    obs = infoIn['taskDesc'] + " OBSERVATION " + obsIn
    return obs


def resetWithVariation(env, variationMin, variationMax, simplificationStr):
    print("ResetWithVariation1")
    variationIdx = random.randrange(variationMin, variationMax)          # train on range 0-20
    #variationIdx = env.getRandomVariationTrain()        ## Random variation on train
    env.reset()    
    print("ResetWithVariation2")
    initialObs, initialDict = env.resetWithVariation(variationIdx, simplificationStr)
    print("Simplifications: " + env.getSimplificationsUsed() )

    print("ResetWithVariation3")
    return initialObs, initialDict

def resetWithVariationTrain(env, simplificationStr):
    print("ResetWithVariation1Train")    
    variationIdx = env.getRandomVariationTrain()        ## Random variation on train
    env.reset()    
    print("ResetWithVariation2Train")
    initialObs, initialDict = env.resetWithVariation(variationIdx, simplificationStr)
    print("Simplifications: " + env.getSimplificationsUsed() )

    print("ResetWithVariation3Train")
    return initialObs, initialDict    

def resetWithVariationTest(env, simplificationStr):
    print("ResetWithVariation1Test")    
    variationIdx = env.getRandomVariationTest()        ## Random variation on test
    env.reset()    
    print("ResetWithVariation2Test")
    initialObs, initialDict = env.resetWithVariation(variationIdx, simplificationStr)
    print("Simplifications: " + env.getSimplificationsUsed() )

    print("ResetWithVariation3Test")
    return initialObs, initialDict    


def reinitializeEnv(threadNum, envStepLimit, taskIdx, variationMin, variationMax, simplificationStr):
    print("ReinitializeEnv1")
    jarPath = "virtualenv-scala-assembly-1.0.jar"
    env = VirtualEnv("", jarPath, envStepLimit, threadNum)      # unusual threadnum so as not to conflict with other running jobs
    taskNames = env.getTaskNames()
    #taskName = taskNames[0]        # Just get first task    
    taskName = taskNames[taskIdx]        # Just get first task    

    #maxVariations = env.getMaxVariations(taskName)
    #variationIdx = random.randrange(variationMin, variationMin)           # Pick a random variation    
    #print("NOTE: Generating random variation: " + str(variationIdx))
    #env.load(taskName, variationIdx)
    #initialObs, initialDict = env.reset() 

    env.load(taskName, 0, simplificationStr)
    initialObs, initialDict = resetWithVariation(env, variationMin, variationMax, simplificationStr)
    print("ReinitializeEnv2")
    return env


######

## TODO, give back-reference to environment through envOLD?
def worker(remote, parent_remote, threadNum, args):
    parent_remote.close()
    #env.create()           # Sciworld - not required?
    print ("------------------------------------ NEW (Thread " + str(threadNum) + ")")

    # Create unique thread
    envStepLimit = args.env_step_limit
    taskIdx = args.task_idx
    variationMin = 0
    variationMax = 100
    env = reinitializeEnv(100+threadNum, envStepLimit, taskIdx, variationMin, variationMax, args.simplification_str)    ## TODO, train/test?

    try:
        done = False
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                #print ("------------------------------------ STEP")
                if done:
                    ##ob, info = env.reset()
                    print("Thread " + str(threadNum) + " is DONE -- resetting with new variation")
                    
                    ob, info = resetWithVariationTrain(env, args.simplification_str)
                    print("Thread " + str(threadNum) + " reset complete")
                    reward = 0
                    done = False
                else:
                    #print("Thread " + str(threadNum) + " step1")
                    ob, reward, done, info = env.step(data)
                    #print("Thread " + str(threadNum) + " step2")

                info = sanitizeInfo(info)
                ob = sanitizeObservation(ob, info)

                #print("Thread " + str(threadNum))
                #print("> " + str(data))
                #print("ob: " + str(ob))                
                #print("score: " + str(info['score']))
                #print("moves: " + str(info['moves']))
                sys.stdout.flush()

                if (done == True):
                    print("DONE -- SCORE " + str(info['score']) + " (Thread " + str(threadNum) + ")")
                    # If we're done, store history in 'info'
                    info['runHistory'] = env.getRunHistory()

                #print("Thread " + str(threadNum) + " step3")                
                remote.send((ob, reward, done, info))
                #print("Thread " + str(threadNum) + " step4")
            elif cmd == 'reset':
                print ("------------------------------------ RESET (Thread " + str(threadNum) + ")")
                ##ob, info = env.reset()
                #print("Thread " + str(threadNum) + " step1A")
                ob, info = resetWithVariationTrain(env, args.simplification_str)                
                #print("Thread " + str(threadNum) + " step1B")
                info = sanitizeInfo(info)
                ob = sanitizeObservation(ob, info)

                #print("Thread " + str(threadNum) + " step1C")
                remote.send((ob, info))
                #print("Thread " + str(threadNum) + " step1D")
            elif cmd == 'get_state':
                print ("------------------------------------ GETSTATE (Thread " + str(threadNum) + ")")
                #print("Thread " + str(threadNum) + " step2A")
                remote.send((env.env.get_state(), done))
                #print("Thread " + str(threadNum) + " step2B")
            elif cmd == 'set_state':
                print ("------------------------------------ SETSTATE (Thread " + str(threadNum) + ")")
                done = data[1]
                #print("Thread " + str(threadNum) + " step3A")
                env.env.set_state(data[0])
                #print("Thread " + str(threadNum) + " step3B")
                remote.send(True)
                #print("Thread " + str(threadNum) + " step3C")
            elif cmd == 'close':
                print ("------------------------------------ CLOSE (Thread " + str(threadNum) + ")")
                #env.close()
                ##env.reset()
                #print("Thread " + str(threadNum) + " step4A")
                env.shutdown()  # Shut down ScienceWorld server for this thread
                time.sleep(2)
                #print("Thread " + str(threadNum) + " step4B")
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        #env.close()
        ##env.reset()
        print ("------------------------------------ SHUTDOWN (Thread " + str(threadNum) + ")")
        env.shutdown()  # Shut down ScienceWorld server for this thread
        time.sleep(2)


class VecEnv:
    def __init__(self, num_envs, programArgs):
        self.closed = False
        #self.env = env
        self.num_envs = num_envs
        self.workerThreadNums = [x for x in range(num_envs)]

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, threadNum, programArgs))
                   for (work_remote, remote, threadNum) in zip(self.work_remotes, self.remotes, self.workerThreadNums)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions):
        self._assert_not_closed()
        assert len(actions) == self.num_envs, "Error: incorrect number of actions."
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        # self.waiting = False
        obs, rewards, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rewards), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return np.stack(obs), infos

    def get_state(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_state', None))

        states = [remote.recv() for remote in self.remotes]
        return states

    def set_state(self, states):
        self._assert_not_closed()
        for remote, state in zip(self.remotes, states):
            remote.send(('set_state', state))

        results = [remote.recv() for remote in self.remotes]
        return results

    def close_extras(self):
        self.closed = True
        for remote in self.remotes:
            remote.send(('close', None))

        time.sleep(5)

        for p in self.ps:
            p.join()

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"
