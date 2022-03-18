import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Pipe
from concurrent.futures import ProcessPoolExecutor as Pool
from env import JerichoEnv


def init_worker(rom_path, step_limit):
    worker.env = JerichoEnv(rom_path, step_limit)
    worker.env.create()


def worker(data):
    (state, done), (cmd, payload) = data

    if cmd != 'reset':
        worker.env.env.set_state(state)

    try:
        if cmd == 'step':
            if done:
                ob, info = worker.env.reset()
                reward, done = 0, False
            else:
                ob, reward, done, info = worker.env.step(payload)

            res = (ob, reward, done, info)
        elif cmd == 'reset':
            ob, info = worker.env.reset()
            res = (ob, info)
        elif cmd == 'close':
            worker.env.close()
            res = None
        else:
            raise NotImplementedError
    except KeyboardInterrupt as e:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
        worker.env.close()
        raise e  # Re-raise

    return (worker.env.env.get_state(), done), res



class VecEnv:
    def __init__(self, num_envs, rom_path, step_limit):
        self.rom_path = rom_path
        self.num_envs = num_envs
        self.states = [(None, False)] * num_envs
        # self.pool = Pool(max_workers=num_envs, initializer=init_worker, initargs=(rom_path, step_limit))
        if not hasattr(VecEnv, 'pool'):
            VecEnv.pool = Pool(initializer=init_worker, initargs=(rom_path, step_limit))
            # VecEnv.pool = mp.Pool(17, initializer=init_worker, initargs=(rom_path, step_limit))

        #self.pool = Pool(initializer=init_worker, initargs=(rom_path, step_limit))

    def step(self, actions):
        assert len(actions) == self.num_envs, "Error: incorrect number of actions."

        args = [(state, ('step', action)) for state, action in zip(self.states, actions)]
        results = self.pool.map(worker, args)

        self.states, results = zip(*results)
        obs, rewards, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rewards), np.stack(dones), infos

    def reset(self):
        args = [(state, ('reset', None)) for state in self.states]
        results = self.pool.map(worker, args)

        self.states, results = zip(*results)
        obs, infos = zip(*results)
        return np.stack(obs), infos

    def get_state(self):
        return self.states

    def set_state(self, states):
        self.states = states

    def __del__(self):
        if hasattr(self, 'pool'):
            self.pool.shutdown()
