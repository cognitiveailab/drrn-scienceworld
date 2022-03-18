from scienceworld import ScienceWorldEnv
from multiprocessing import Process, Pipe

import numpy as np
import random
import time
import sys

#
#   Input/Output sanitization
#
def sanitizeInfo(infoIn):
    # Convert from py4j.java_collections.JavaList to python list
    recastList = []
    for elem in infoIn['valid']:
        recastList.append(elem)

    info = {'moves': infoIn['moves'],
            'score': infoIn['score'],
            'look': infoIn['look'],
            'inv': infoIn['inv'],            
            'valid': recastList,
            'taskDesc': infoIn['taskDesc']
        }

    return info


def sanitizeObservation(obsIn, infoIn):
    obs = infoIn['taskDesc'] + " OBSERVATION " + obsIn
    return obs

#
# Reset the environment (with a new randomly selected variation)
#
def resetWithVariation(env, variationMin, variationMax, simplificationStr):    
    variationIdx = random.randrange(variationMin, variationMax)          # train on range 0-20    
    env.reset()        
    initialObs, initialDict = env.resetWithVariation(variationIdx, simplificationStr)
    print("Simplifications: " + env.getSimplificationsUsed() )
    
    return initialObs, initialDict

def resetWithVariationTrain(env, simplificationStr):
    variationIdx = env.getRandomVariationTrain()        ## Random variation on train
    env.reset()    
    initialObs, initialDict = env.resetWithVariation(variationIdx, simplificationStr)
    print("Simplifications: " + env.getSimplificationsUsed() )

    return initialObs, initialDict    

def resetWithVariationDev(env, simplificationStr):
    variationIdx = env.getRandomVariationDev()          ## Random variation on dev
    env.reset()        
    initialObs, initialDict = env.resetWithVariation(variationIdx, simplificationStr)
    print("Simplifications: " + env.getSimplificationsUsed() )
    
    return initialObs, initialDict    

def resetWithVariationTest(env, simplificationStr):    
    variationIdx = env.getRandomVariationTest()        ## Random variation on test
    env.reset()        
    initialObs, initialDict = env.resetWithVariation(variationIdx, simplificationStr)
    print("Simplifications: " + env.getSimplificationsUsed() )
    
    return initialObs, initialDict    

# Initialize a ScienceWorld environment directly from the API
def initializeEnv(threadNum, args):    
    env = ScienceWorldEnv("", None, args.env_step_limit, threadNum)

    taskNames = env.getTaskNames()    
    taskName = taskNames[args.task_idx]

    # Just reset to variation 0, as another call (e.g. resetWithVariation...) will setup an appropriate variation (train/dev/test)
    env.load(taskName, 0, args.simplification_str)
    initialObs, initialDict = resetWithVariation(env, 0, 1, args.simplification_str)

    return env




#
#   Worker
#
def worker(remote, parent_remote, threadNum, args):
    parent_remote.close()
    print ("------------------------------------ NEW (Thread " + str(threadNum) + ")")

    # Create unique thread
    # Note, it doesn't matter what the variation is initially initialized to -- we reset it to a proper variation # (train/dev/test) before use. 
    env = initializeEnv(threadNum = 100+threadNum, args=args)

    try:
        done = False
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':                
                if done:                    
                    # If the thread is done, reset it
                    print("Thread " + str(threadNum) + " is DONE -- resetting with new variation")                    
                    ob, info = resetWithVariationTrain(env, args.simplification_str)
                    print("Thread " + str(threadNum) + " reset complete")
                    reward = 0
                    done = False
                else:                    
                    # Otherwise, complete one step
                    ob, reward, done, info = env.step(data)
                    
                # Sanitize the 'observation' and 'info' receieved from the API
                info = sanitizeInfo(info)
                ob = sanitizeObservation(ob, info)

                # Make sure any stdout from this thread is printed to the console in a timely fashion
                sys.stdout.flush()

                if (done == True):
                    print("DONE -- SCORE " + str(info['score']) + " (Thread " + str(threadNum) + ")")
                    # If we're done, store history in 'info'
                    info['runHistory'] = env.getRunHistory()
                
                remote.send((ob, reward, done, info))
                
            elif cmd == 'reset':
                print ("------------------------------------ RESET (Thread " + str(threadNum) + ")")
                ob, info = resetWithVariationTrain(env, args.simplification_str)                
                info = sanitizeInfo(info)
                ob = sanitizeObservation(ob, info)
                
                remote.send((ob, info))

            elif cmd == 'get_state':
                print ("------------------------------------ GETSTATE (Thread " + str(threadNum) + ")")                
                remote.send((env.env.get_state(), done))
                

            elif cmd == 'set_state':
                print ("------------------------------------ SETSTATE (Thread " + str(threadNum) + ")")
                done = data[1]
                env.env.set_state(data[0])                

                remote.send(True)
                
            elif cmd == 'close':
                print ("------------------------------------ CLOSE (Thread " + str(threadNum) + ")")
                env.shutdown()  # Shut down ScienceWorld server for this thread
                time.sleep(2)
                break

            else:
                raise NotImplementedError

    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        print ("------------------------------------ SHUTDOWN (Thread " + str(threadNum) + ")")
        env.shutdown()  # Shut down ScienceWorld server for this thread
        time.sleep(2)


#
#   VecEnv: Handles spawning up 'num_envs' workers.
#
class VecEnv:
    def __init__(self, num_envs, programArgs):
        self.closed = False        
        self.num_envs = num_envs
        self.workerThreadNums = [x for x in range(num_envs)]        # A different thread number (0-numEnvs) for each worker thread, so the ScienceWorld servers spawn on different ports

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
