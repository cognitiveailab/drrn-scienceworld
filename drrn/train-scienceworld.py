import time
import timeit
import torch
import logger
import argparse
from drrn import DRRN_Agent
import numpy as np

import gymnasium as gym
from gymnasium.wrappers import TimeLimit, AutoResetWrapper
from functools import partial

from scienceworld import ScienceWorldEnv, BufferedHistorySaver
from vec_env import resetWithVariation, resetWithVariationDev, resetWithVariationTest, initializeEnv, sanitizeInfo, sanitizeObservation


def configure_logger(log_dir):
    logger.configure(log_dir, format_strs=['log'])
    global tb
    tb = logger.Logger(log_dir, [#logger.make_output_format('tensorboard', log_dir),
                                 logger.make_output_format('csv', log_dir),
                                 #logger.make_output_format('stdout', log_dir)
                                 ])
    global log
    log = logger.log

# def clean(strIn):
#     charsToFilter = ['\t', '\n', '*', '-']
#     for c in charsToFilter:
#         strIn = strIn.replace(c, ' ')
#     return strIn.strip()


def evaluate(agent, envs_eval, options_eval, args, bufferedHistorySaverEval, extraSaveInfo):

    rng = np.random.default_rng(args.seed)
    seeds = list(map(int, rng.integers(2**32, size=envs_eval.num_envs)))
    with torch.no_grad():
        obs, infos = envs_eval.reset(seed=seeds, options=options_eval)
        dones = np.array([False] * envs_eval.num_envs)

        #log("Starting evaluation")
        while not np.all(dones):
            # Encode state and valid actions.
            states = agent.build_state(obs, infos)
            valid_ids = [agent.encode(valid) for valid in infos['valid']]

            # Choose actions
            action_ids, action_idxs, _ = agent.act(states, valid_ids, sample=False)
            action_strs = [valid[idx] for valid, idx in zip(infos['valid'], action_idxs)]

            # Perform the actions in the environments
            obs, rewards, terminateds, truncateds, infos = envs_eval.step(action_strs)
            dones = np.logical_or(terminateds, truncateds)

        print("Completed evaluation")
        for runHistory, evalIdx in zip(infos["runHistory"], infos["variationIdx"]):
            episodeIdx = str(extraSaveInfo["stepsFunctional"]) + "-" + str(evalIdx)
            bufferedHistorySaverEval.storeRunHistory(runHistory, episodeIdx, notes=dict(extraSaveInfo))
            bufferedHistorySaverEval.saveRunHistoriesBufferIfFull(maxPerFile=extraSaveInfo['maxHistoriesPerFile'])

        avg_score = np.mean(infos['score'])
        return infos['score'], avg_score


class SkipDone(gym.Wrapper):

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._last_state = None
        self._is_done = False
        return observation, info

    def step(self, action):
        if not self._is_done:
            observation, reward, terminated, truncated, info = self.env.step(action)
            self._is_done = terminated or truncated
            info["_terminated"] = terminated
            info["_truncated"] = truncated

            terminated = truncated = False  # To avoid being autoreset.
            self._last_state = (observation, reward, terminated, truncated, info)

        return self._last_state

class PostProcessSkipDone(gym.vector.VectorWrapper):

    def step(self, actions):
        observations, rewards, terminateds, truncateds, infos = self.env.step(actions)
        terminateds = infos["_terminated"]
        truncateds = infos["_truncated"]
        return observations, rewards, terminateds, truncateds, infos


class SanitizeScienceWorld(gym.Wrapper):

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = info['taskDesc'] + " OBSERVATION " + obs
        info["runHistory"] = ""
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = info['taskDesc'] + " OBSERVATION " + obs
        info["runHistory"] = ""
        if terminated or truncated:
            info["runHistory"] = self.env.unwrapped.env.getRunHistory()

        return obs, reward, terminated, truncated, info


def train(agent, envs, max_steps, update_freq, eval_freq, checkpoint_freq, log_freq, args, bufferedHistorySaverTrain, bufferedHistorySaverEval):
    startTime = time.time()
    flush_cache_freq = 100

    #max_steps = int(math.ceil(max_steps / args.num_envs))
    numEpisodes = 0
    stepsFunctional = 0
    start1 = timeit.default_timer()

    # Initialize a threaded wrapper for the ScienceWorld environment
    TimeLimitWrapper = partial(TimeLimit, max_episode_steps=args.env_step_limit)

    envs_train = gym.make_vec(
        "ScienceWorld-v0",
        num_envs=args.num_envs,
        vectorization_mode="async",
        vector_kwargs={"shared_memory": False},
        wrappers=[TimeLimitWrapper, SanitizeScienceWorld]
    )
    envs_eval = gym.make_vec(
        "ScienceWorld-v0",
        num_envs=10,
        vectorization_mode="async",
        vector_kwargs={"shared_memory": False},
        wrappers=[TimeLimitWrapper, SanitizeScienceWorld, SkipDone]
    )
    envs_eval = PostProcessSkipDone(envs_eval)

    options_train = {
        "task": args.task_idx,
        "variation": "train",
        "simplification": args.simplification_str,
    }
    options_eval = {
        "task": args.task_idx,
        "variation": args.eval_set,
        "simplification": args.simplification_str,
    }

    # Reinit environments
    rng = np.random.default_rng(args.seed)
    seeds = list(map(int, rng.integers(2**32, size=envs_train.num_envs)))
    obs, infos = envs_train.reset(seed=seeds, options=options_train)

    states = agent.build_state(obs, infos)
    valid_ids = [agent.encode(valid) for valid in infos['valid']]
    loss = np.inf
    for step in range(1, max_steps+1):
        stepsFunctional = step * envs_train.num_envs

        # Summary statistics
        #print("-------------------")
        end = timeit.default_timer()
        deltaTime = end - start1
        deltaTimeMins = deltaTime / 60
        print(f"Step {step}. Loss: {loss:.4f} ({deltaTimeMins:.2f} minutes)")

        # Choose action(s)
        action_ids, action_idxs, _ = agent.act(states, valid_ids)
        action_strs = [valid[idx] for valid, idx in zip(infos['valid'], action_idxs)]

        # Perform the action(s) in the environment
        obs, rewards, terminateds, truncateds, infos = envs_train.step(action_strs)
        dones = np.logical_or(terminateds, truncateds)

        # Check for any completed episodes
        for i, (done, score) in enumerate(zip(dones, infos['score'])):
            if done:
                # An episode has completed
                tb.logkv('EpisodeScore', score)
                print("EPISODE SCORE: " + str(score) + " STEPS: " + str(step) + " STEPS (functional): " + str(stepsFunctional) + " EPISODES: " + str(numEpisodes))

                # Save the environment's history in the history logs
                bufferedHistorySaverTrain.storeRunHistory(infos["final_info"][i]["runHistory"], numEpisodes, notes={'step':step})
                bufferedHistorySaverTrain.saveRunHistoriesBufferIfFull(maxPerFile=args.maxHistoriesPerFile)

                numEpisodes += 1

        next_states = agent.build_state(obs, infos)
        next_valids = [agent.encode(valid) for valid in infos['valid']]
        for state, act, rew, next_state, valids, done in \
            zip(states, action_ids, rewards, next_states, next_valids, dones):
            agent.observe(state, act, rew, next_state, valids, done)
        states = next_states
        valid_ids = next_valids

        if step % log_freq == 0:
            tb.logkv('Step', step)
            tb.logkv('StepsFunctional', stepsFunctional)
            tb.logkv("FPS", int((stepsFunctional)/(time.time()-startTime)))
            tb.logkv('numEpisodes', numEpisodes)
            tb.logkv('taskIdx', args.task_idx)
            tb.logkv('GPU_mem', agent.getMemoryUsage())

            print("*************************")
            print("Step:            " + str(step))
            print("StepsFunctional: " + str(stepsFunctional))
            print("FPS:             " + str(stepsFunctional/(time.time()-startTime)) )
            print("numEpisodes:     " + str(numEpisodes))
            print("taskIdx:         " + str(args.task_idx))
            print("GPU_mem:         " + str(agent.getMemoryUsage()))
            print("*************************")

        if step % update_freq == 0:
            loss = agent.update()
            if loss is not None:
                tb.logkv_mean('Loss', loss)
            else:
                loss = np.inf

        if step % checkpoint_freq == 0:
            # Save model checkpoints
            agent.save("-steps" + str(stepsFunctional) + "-eps" + str(numEpisodes))

        if step % flush_cache_freq == 0:
            # Keep the GPU memory low
            agent.clearGPUCache()

        if step % eval_freq == 0:
            # Do the evaluation procedure
            extraSaveInfo = {'numEpisodes':numEpisodes, 'numSteps':step, 'stepsFunctional':stepsFunctional, 'maxHistoriesPerFile':args.maxHistoriesPerFile}
            eval_scores, avg_eval_score = evaluate(agent, envs_eval, options_eval, args, bufferedHistorySaverEval, extraSaveInfo)

            tb.logkv('EvalScore', avg_eval_score)
            tb.logkv('numEpisodes', numEpisodes)
            tb.dumpkvs()

            for eval_score in eval_scores:
                print("EVAL EPISODE SCORE: " + str(eval_score) + " STEPS: " + str(step) + " STEPS (functional): " + str(stepsFunctional) + " EPISODES: " + str(numEpisodes))

            envs_train.reset()


    # Save anything left in history buffers
    bufferedHistorySaverTrain.saveRunHistoriesBufferIfFull(maxPerFile=args.maxHistoriesPerFile, forceSave=True)
    bufferedHistorySaverEval.saveRunHistoriesBufferIfFull(maxPerFile=args.maxHistoriesPerFile, forceSave=True)


    print("Training complete.")
    # Final save
    agent.save("-steps" + str(stepsFunctional) + "-eps" + str(numEpisodes))

    # Close environments
    #envs.close_extras()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='logs')
    parser.add_argument('--spm_path', default='../spm_models/unigram_8k.model')
    parser.add_argument('--env_step_limit', default=100, type=int)
    parser.add_argument('--seed', default=20221108, type=int)
    parser.add_argument('--num_envs', default=8, type=int)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--checkpoint_freq', default=5000, type=int)
    parser.add_argument('--eval_freq', default=1000, type=int)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--memory_size', default=100000, type=int)
    parser.add_argument('--priority_fraction', default=0.5, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gamma', default=.9, type=float)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--clip', default=5, type=float)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)

    parser.add_argument('--task_idx', default=0, type=int)
    parser.add_argument('--maxHistoriesPerFile', default=1000, type=int)
    parser.add_argument('--historySavePrefix')

    parser.add_argument('--eval_set', default='test', type=str)      # 'dev' or 'test'

    parser.add_argument('--simplification_str', default='easy', type=str)

    return parser.parse_args()



def main():
    args = parse_args()
    print(args)
    configure_logger(args.output_dir)
    agent = DRRN_Agent(args)

    # Initialize the save buffers
    taskIdx = args.task_idx
    history_save_prefix = args.historySavePrefix or args.output_dir
    bufferedHistorySaverTrain = BufferedHistorySaver(filenameOutPrefix=f"{history_save_prefix}/history-seed{args.seed}-task{taskIdx}-train")
    bufferedHistorySaverEval = BufferedHistorySaver(filenameOutPrefix=f"{history_save_prefix}/history-seed{args.seed}-task{taskIdx}-{args.eval_set}")

    # Start training
    start = timeit.default_timer()

    train(agent, None, args.max_steps, args.update_freq, args.eval_freq,
          args.checkpoint_freq, args.log_freq, args, bufferedHistorySaverTrain, bufferedHistorySaverEval)

    end = timeit.default_timer()
    deltaTime = end - start
    deltaTimeMins = deltaTime / 60
    print("Runtime: " + str(deltaTime) + " seconds  (" + str(deltaTimeMins) + " minutes)")

    print("Rate: " + str(args.max_steps / deltaTime) + " steps/second")
    print("SimplificationStr: " + str(args.simplification_str))



if __name__ == "__main__":
    main()
