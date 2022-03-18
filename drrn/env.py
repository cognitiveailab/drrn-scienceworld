import time
from logging import exception

import redis

from jericho import FrotzEnv
from jericho.util import *
from jericho.defines import *


def load_vocab_rev(env):
    vocab = {i+2: str(v) for i, v in enumerate(env.get_dictionary())}
    vocab[0] = ' '
    vocab[1] = '<s>'
    vocab_rev = {v: idx for idx, v in vocab.items()}
    return vocab_rev


class JerichoEnv:
    ''' Returns valid actions at each step of the game. '''
    def __init__(self, rom_path, step_limit=None):
        self.rom_path = rom_path
        self.steps = 0
        self.step_limit = step_limit
        self.env = None
        self.conn = None
        self.vocab_rev = None

    def create(self):
        self.env = FrotzEnv(self.rom_path)
        self.vocab_rev = load_vocab_rev(self.env)
        self.conn = redis.Redis(host='localhost', port=6379, db=0)

        while True:
            try:
                if self.conn.ping():
                    break
            except redis.exceptions.BusyLoadingError:
                # print(f"[{os.getpid()}] waiting for Redis to load dataset in memory.")
                time.sleep(1)

        print(f"[{os.getpid()}] Connection to Redis established.")

    def _gather_info(self):
        info = {}

        env_ = self.env.copy()
        look, _, _, _ = env_.step('look')
        info['look'] = look

        env_ = self.env.copy()
        inv, _, _, _ = env_.step('inventory')
        info['inv'] = inv

        info['valid'], info['cache_hit'] = self._get_valid_actions()
        if len(info['valid']) == 0:
            info['valid'] = ['wait','yes','no']

        return info

    def _get_valid_actions(self):
        """ Get the valid actions for the current state. """
        valid_actions = []

        # SPEED: Avoid calling self.env.get_valid_actions for already visited states.
        world_state_hash = self.env.get_world_state_hash()
        res = self.conn.get(world_state_hash)
        if res is None:
            # valid_actions = self.env.get_valid_actions(use_ctypes=True)  # Bottleneck.
            valid_actions = self.env.get_valid_actions(use_ctypes=True, use_parallel=True)  # Bottleneck.
            # valid_actions = self.env.get_valid_actions_c()  # Bottleneck.
            self.conn.set(world_state_hash, '/'.join(valid_actions))
        else:
            res = res.decode('cp1252')
            if res:
                valid_actions = res.split('/')

        cache_hit = res is not None
        return valid_actions, cache_hit

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        # Initialize with default values
        info['look'] = 'unknown'
        info['inv'] = 'unknown'
        info['valid'] = ['wait','yes','no']
        info['cache_hit'] = True
        if not done:
            info.update(self._gather_info())

        self.steps += 1
        if self.step_limit and self.steps >= self.step_limit:
            done = True

        return ob, reward, done, info

    def reset(self):
        self.steps = 0
        initial_ob, info = self.env.reset()
        info.update(self._gather_info())
        return initial_ob, info

    def get_dictionary(self):
        if not self.env:
            self.create()

        return self.env.get_dictionary()

    def get_action_set(self):
        return None

    def close(self):
        self.env.close()
