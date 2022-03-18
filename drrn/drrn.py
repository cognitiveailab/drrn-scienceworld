import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join as pjoin
from memory import ReplayMemory, PrioritizedReplayMemory, Transition, State
from model import DRRN
from util import *
import logger
import sentencepiece as spm

import shutil
import os
import sys
import time

import signal
from contextlib import contextmanager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#
#   Timeout additions (from https://www.jujens.eu/posts/en/2018/Jun/02/python-timeout-function/ )
#
@contextmanager
def timeout(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)

    try:
        yield
    except TimeoutError:
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def raise_timeout(signum, frame):
    print("Timeout")
    raise TimeoutError

#
#   DRRN
#
class DRRN_Agent:
    def __init__(self, args):
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(args.spm_path)
        self.network = DRRN(len(self.sp), args.embedding_dim, args.hidden_dim).to(device)
        ## self.memory = ReplayMemory(args.memory_size)     ## PJ: Changing to more memory efficient memory, since the pickle files are enormous
        self.memory = PrioritizedReplayMemory(capacity = args.memory_size, priority_fraction = args.priority_fraction)     ## PJ: Changing to more memory efficient memory, since the pickle files are enormous
        self.save_path = args.output_dir
        self.clip = args.clip
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=args.learning_rate)

        # Error catching when saving the model
        self.lastSaveSuccessful = True
        self.numSaveErrors = 0

    def observe(self, state, act, rew, next_state, next_acts, done):
        #self.memory.push(state, act, rew, next_state, next_acts, done)     # When using ReplayMemory
        self.memory.push(False, state, act, rew, next_state, next_acts, done)       # When using PrioritizedReplayMemory (? PJ)


    def build_state(self, obs, infos):
        """ Returns a state representation built from various info sources. """
        obs_ids = [self.sp.EncodeAsIds(o) for o in obs]
        # TextWorld
        look_ids = [self.sp.EncodeAsIds(info['look']) for info in infos]
        inv_ids = [self.sp.EncodeAsIds(info['inv']) for info in infos]
        # ScienceWorld

        #print("obs:")
        #print(obs)
        #print("infos:")
        #print(infos)        
        #look_ids = [self.sp.EncodeAsIds(info['look']) for info in infos]
        #inv_ids = [self.sp.EncodeAsIds(info['inv']) for info in infos]

        return [State(ob, lk, inv) for ob, lk, inv in zip(obs_ids, look_ids, inv_ids)]


    def encode(self, obs_list):
        """ Encode a list of observations """
        return [self.sp.EncodeAsIds(o) for o in obs_list]


    def act(self, states, poss_acts, sample=True):
        """ Returns a string action from poss_acts. """
        idxs, values = self.network.act(states, poss_acts, sample)
        act_ids = [poss_acts[batch][idx] for batch, idx in enumerate(idxs)]
        return act_ids, idxs, values


    def update(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute Q(s', a') for all a'
        # TODO: Use a target network???
        next_qvals = self.network(batch.next_state, batch.next_acts)
        # Take the max over next q-values
        next_qvals = torch.tensor([vals.max() for vals in next_qvals], device=device)
        # Zero all the next_qvals that are done
        next_qvals = next_qvals * (1-torch.tensor(batch.done, dtype=torch.float, device=device))
        targets = torch.tensor(batch.reward, dtype=torch.float, device=device) + self.gamma * next_qvals

        # Next compute Q(s, a)
        # Nest each action in a list - so that it becomes the only admissible cmd
        nested_acts = tuple([[a] for a in batch.act])
        qvals = self.network(batch.state, nested_acts)
        # Combine the qvals: Maybe just do a greedy max for generality
        qvals = torch.cat(qvals)

        # Compute Huber loss
        loss = F.smooth_l1_loss(qvals, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.clip)
        self.optimizer.step()
        return loss.item()


    def load(self, path, suffixStr=""):
        print("Loading agent from path: " + str(path))
        try:
            self.memory = pickle.load(open(pjoin(path, "memory" + suffixStr + ".pkl"), 'rb'))
            self.network = torch.load(pjoin(path, "model" + suffixStr + ".pt"))
        except Exception as e:
            print("Error loading model.")
            logging.error(traceback.format_exc())


    def save(self, suffixStr=""):
        startTime = time.time()
        print("Saving agent to path: " + str(self.save_path))
        print("Started saving at: " + str(startTime))
        sys.stdout.flush()
        
        # First, remove any old backups
        print("Removing old backups")
        sys.stdout.flush()
        try: 
            files = os.listdir(self.save_path + "/bak")
            for filename in files:
                if (filename.startswith("memory")) or (filename.startswith("model") or (filename.startswith("progress") or (filename.startswith("log")))):
                    os.remove(self.save_path + "/bak/" + filename)
        except Exception as e:
            print("Error removing backups.")


        try:
            # First, copy old backups
            if (self.lastSaveSuccessful == True):
                print("Creating backups")
                sys.stdout.flush()
                os.makedirs(self.save_path + "/bak", exist_ok=True)
                files = os.listdir(self.save_path)
                for filename in files:
                    if filename.startswith("memory") or filename.startswith("model"):                
                        shutil.move(self.save_path + "/" + filename, self.save_path + "/bak/" + filename)
                    if filename.startswith("progress") or filename.startswith("log"):                
                        shutil.copy(self.save_path + "/" + filename, self.save_path + "/bak/" + filename)


            print("Saving files")
            sys.stdout.flush()
            # Then, save new ones
            #print("JSON Serialization test")
            #self.memory.serializeToJSON(self.save_path + "/" + "memory.json")

            self.lastSaveSuccessful = False
            with timeout(120):
                print("Pickle")            
                print("Length: " + str(len(self.memory)) )
                sys.stdout.flush()
                pickle.dump(self.memory, open(pjoin(self.save_path, "memory" + str(suffixStr) + ".pkl"), 'wb'))
                print("Torch")
                sys.stdout.flush()
                torch.save(self.network, pjoin(self.save_path, "model" + str(suffixStr) + ".pt"))
                print("Done")
                success = True
                self.lastSaveSuccessful = True

            if (self.lastSaveSuccessful == False):
                print("* Model failed to save (timeout).")
                self.numSaveErrors += 1
            
            print("Total number of save timeouts since running: " + str(self.numSaveErrors))

            sys.stdout.flush()

        except Exception as e:
            print("Error saving model.")
            logging.error(traceback.format_exc())

        print("Total save time: " + str(time.time()-startTime))
        sys.stdout.flush()

    # PJ
    def getMemoryUsage(self):
        usage = torch.cuda.memory_allocated(device=device)/1024./1024.
        reserved = torch.cuda.memory_reserved(device=device)/1024./1024.
        print ('GPU Memory Allocated {} MB'.format(usage))
        return usage

    def clearGPUCache(self):
        print ("Clearing GPU Cache...")
        torch.cuda.empty_cache()