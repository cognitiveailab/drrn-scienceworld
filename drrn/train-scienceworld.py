import subprocess
import time
import math
import timeit
import torch
import logger
import argparse
from drrn import DRRN_Agent
from vec_env import VecEnv
import random

from scienceworld import ScienceWorldEnv, BufferedHistorySaver
from vec_env import resetWithVariation, resetWithVariationTrain, resetWithVariationTest, initializeEnv


def configure_logger(log_dir):
    logger.configure(log_dir, format_strs=['log'])
    global tb
    tb = logger.Logger(log_dir, [logger.make_output_format('tensorboard', log_dir),
                                 logger.make_output_format('csv', log_dir),
                                 logger.make_output_format('stdout', log_dir)])
    global log
    log = logger.log

def clean(strIn):
    charsToFilter = ['\t', '\n', '*', '-']
    for c in charsToFilter:
        strIn = strIn.replace(c, ' ')
    return strIn.strip()


def evaluate(agent, args, env_step_limit, bufferedHistorySaverEval, extraSaveInfo, nb_episodes=10):    
    # Initialize a ScienceWorld thread for serial evaluation
    env = initializeEnv(threadNum = args.num_envs+10, args=args) # A threadNum (and therefore port) that shouldn't be used by any of the regular training workers

    scoresOut = []
    with torch.no_grad():
        
        for ep in range(nb_episodes):
            total_score = 0
            log("Starting evaluation episode {}".format(ep))   
            print("Starting evaluation episode " + str(ep) + " / " + str(nb_episodes))         
            extraSaveInfo['evalIdx'] = ep
            score = evaluate_episode(agent, env, env_step_limit, args.simplification_str, bufferedHistorySaverEval, extraSaveInfo, args.eval_set)
            log("Evaluation episode {} ended with score {}\n\n".format(ep, score))
            total_score += score
            scoresOut.append(total_score)
            print("")

        avg_score = total_score / nb_episodes
        
        env.shutdown()
        
        return scoresOut, avg_score

    


def evaluate_episode(agent, env, env_step_limit, simplificationStr, bufferedHistorySaverEval, extraSaveInfo, evalSet):
    step = 0
    done = False
    numSteps = 0
    ob = ""
    info = {}
    if (evalSet == "dev"):
        ob, info = resetWithVariationTest(env, simplificationStr)
    elif (evalSet == "test"):
        ob, info = resetWithVariationTest(env, simplificationStr)
    else:
        print("evaluate_episode: unknown evaluation set (expected 'dev' or 'test', found: " + str(evalSet) + ")")
        env.shutdown()

        exit(1)


    state = agent.build_state([ob], [info])[0]
    log('Obs{}: {} Inv: {} Desc: {}'.format(step, clean(ob), clean(info['inv']), clean(info['look'])))    
    while not done:
        #print("numSteps: " + str(numSteps))        
        valid_acts = info['valid']
        valid_ids = agent.encode(valid_acts)        
        _, action_idx, action_values = agent.act([state], [valid_ids], sample=False)        
        action_idx = action_idx[0]
        action_values = action_values[0]
        action_str = valid_acts[action_idx]
        log('Action{}: {}, Q-Value {:.2f}'.format(step, action_str, action_values[action_idx].item()))        
        s = ''

        maxToDisplay = 10   # Max Q values to display, to limit the log size
        numDisplayed = 0
        for idx, (act, val) in enumerate(sorted(zip(valid_acts, action_values), key=lambda x: x[1], reverse=True), 1):
            s += "{}){:.2f} {} ".format(idx, val.item(), act)
            numDisplayed += 1
            if (numDisplayed > maxToDisplay):
                break

        log('Q-Values: {}'.format(s))
        ob, rew, done, info = env.step(action_str)
        
        log("Reward{}: {}, Score {}, Done {}".format(step, rew, info['score'], done))        
        step += 1
        log('Obs{}: {} Inv: {} Desc: {}'.format(step, clean(ob), clean(info['inv']), clean(info['look'])))
        state = agent.build_state([ob], [info])[0]        

        numSteps +=1        
        if (numSteps > env_step_limit):
            print("Maximum number of evaluation steps reached (" + str(env_step_limit) + ").")
            break    

    print("Completed one evaluation episode")
    # Save
    runHistory = env.getRunHistory()
    episodeIdx = str(extraSaveInfo['numEpisodes']) + "-" + str(extraSaveInfo['evalIdx'])
    bufferedHistorySaverEval.storeRunHistory(runHistory, episodeIdx, notes=extraSaveInfo)
    bufferedHistorySaverEval.saveRunHistoriesBufferIfFull(maxPerFile=extraSaveInfo['maxHistoriesPerFile'])
    print("Completed saving")


    return info['score']


def train(agent, envs, max_steps, update_freq, eval_freq, checkpoint_freq, log_freq, args, bufferedHistorySaverTrain, bufferedHistorySaverEval):
    startTime = time.time()
    flush_cache_freq = 100

    #max_steps = int(math.ceil(max_steps / args.num_envs))
    numEpisodes = 0
    stepsFunctional = 0
    start1 = timeit.default_timer()


    # Reinit environments
    obs, infos = envs.reset()

    states = agent.build_state(obs, infos)
    valid_ids = [agent.encode(info['valid']) for info in infos]
    for step in range(1, max_steps+1):
        stepsFunctional = step * envs.num_envs

        # Summary statistics
        print("-------------------")
        print("Step " + str(step))
        print("")
        end = timeit.default_timer()
        deltaTime = end - start1
        deltaTimeMins = deltaTime / 60
        print("Started at runtime: " + str(deltaTime) + " seconds  (" + str(deltaTimeMins) + " minutes)")
        print("")

        # Choose action(s)
        action_ids, action_idxs, _ = agent.act(states, valid_ids)
        action_strs = [info['valid'][idx] for info, idx in zip(infos, action_idxs)]        

        # Perform the action(s) in the environment
        obs, rewards, dones, infos = envs.step(action_strs)

        # Check for any completed episodes
        for done, info in zip(dones, infos):
            if done:
                # An episode has completed
                tb.logkv('EpisodeScore', info['score'])
                print("EPISODE SCORE: " + str(info['score']))
                print("EPISODE SCORE: " + str(info['score']) + " STEPS: " + str(step) + " STEPS (functional): " + str(stepsFunctional) + " EPISODES: " + str(numEpisodes))

                # Save the environment's history in the history logs
                runHistory = info['runHistory']
                bufferedHistorySaverTrain.storeRunHistory(runHistory, numEpisodes, notes={'step':step})
                bufferedHistorySaverTrain.saveRunHistoriesBufferIfFull(maxPerFile=args.maxHistoriesPerFile)

                numEpisodes += 1

        next_states = agent.build_state(obs, infos)
        next_valids = [agent.encode(info['valid']) for info in infos]        
        for state, act, rew, next_state, valids, done in \
            zip(states, action_ids, rewards, next_states, next_valids, dones):
            agent.observe(state, act, rew, next_state, valids, done)
        states = next_states
        valid_ids = next_valids

        if step % log_freq == 0:            
            tb.logkv('Step', step)
            tb.logkv('StepsFunctional', step*envs.num_envs)
            tb.logkv("FPS", int((step*envs.num_envs)/(time.time()-startTime)))
            tb.logkv('numEpisodes', numEpisodes)
            tb.logkv('taskIdx', args.task_idx)
            tb.logkv('GPU_mem', agent.getMemoryUsage())

            print("*************************")
            print("Step:            " + str(step))
            print("StepsFunctional: " + str(step*envs.num_envs))
            print("FPS:             " + str( (step*envs.num_envs)/(time.time()-startTime)) )
            print("numEpisodes:     " + str(numEpisodes))
            print("taskIdx:         " + str(args.task_idx))
            print("GPU_mem:         " + str(agent.getMemoryUsage()))
            print("*************************")

        if step % update_freq == 0:            
            loss = agent.update()            
            if loss is not None:
                tb.logkv_mean('Loss', loss)

        if step % checkpoint_freq == 0:
            # Save model checkpoints
            agent.save("-steps" + str(stepsFunctional) + "-eps" + str(numEpisodes))

        if step % flush_cache_freq == 0:
            # Keep the GPU memory low
            agent.clearGPUCache()

        if step % eval_freq == 0:
            # Do the evaluation procedure
            extraSaveInfo = {'numEpisodes':numEpisodes, 'numSteps':step, 'stepsFunctional:':stepsFunctional, 'maxHistoriesPerFile':args.maxHistoriesPerFile}
            eval_scores, avg_eval_score = evaluate(agent, args, args.env_step_limit, bufferedHistorySaverEval, extraSaveInfo)
            
            tb.logkv('EvalScore', avg_eval_score)
            tb.logkv('numEpisodes', numEpisodes)
            tb.dumpkvs()

            for eval_score in eval_scores:
                print("EVAL EPISODE SCORE: " + str(eval_score))
                print("EVAL EPISODE SCORE: " + str(eval_score) + " STEPS: " + str(step) + " STEPS: " + str(stepsFunctional) + " EPISODES: " + str(numEpisodes))

            envs.reset()


    # Save anything left in history buffers
    bufferedHistorySaverTrain.saveRunHistoriesBufferIfFull(maxPerFile=args.maxHistoriesPerFile, forceSave=True)
    bufferedHistorySaverEval.saveRunHistoriesBufferIfFull(maxPerFile=args.maxHistoriesPerFile, forceSave=True)


    print("Training complete.")
    # Final save
    agent.save("-steps" + str(stepsFunctional) + "-eps" + str(numEpisodes))
    # Close environments
    envs.close_extras()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='logs')
    parser.add_argument('--spm_path', default='../spm_models/unigram_8k.model')
    parser.add_argument('--rom_path', default='zork1.z5')
    parser.add_argument('--env_step_limit', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_envs', default=16, type=int)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--checkpoint_freq', default=500, type=int)
    parser.add_argument('--eval_freq', default=500, type=int)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--memory_size', default=5000000, type=int)
    parser.add_argument('--priority_fraction', default=0.0, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gamma', default=.9, type=float)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--clip', default=5, type=float)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)

    parser.add_argument('--task_idx', default=0, type=int)    
    parser.add_argument('--maxHistoriesPerFile', default=1000, type=int)
    parser.add_argument('--historySavePrefix', default='saveout', type=str)

    parser.add_argument('--eval_set', default='dev', type=str)      # 'dev' or 'test'

    parser.add_argument('--simplification_str', default='', type=str)

    return parser.parse_args()



def main():
    ## assert jericho.__version__ == '2.1.0', "This code is designed to be run with Jericho version 2.1.0."
    args = parse_args()
    print(args)
    configure_logger(args.output_dir)    
    agent = DRRN_Agent(args)

    # Initialize a threaded wrapper for the ScienceWorld environment
    envs = VecEnv(args.num_envs, args)

    # Initialize the save buffers
    taskIdx = args.task_idx
    bufferedHistorySaverTrain = BufferedHistorySaver(filenameOutPrefix = args.historySavePrefix + "-task" + str(taskIdx) + "-train")
    bufferedHistorySaverEval = BufferedHistorySaver(filenameOutPrefix = args.historySavePrefix + "-task" + str(taskIdx) + "-eval")

    # Start training
    start = timeit.default_timer()

    train(agent, envs, args.max_steps, args.update_freq, args.eval_freq,
          args.checkpoint_freq, args.log_freq, args, bufferedHistorySaverTrain, bufferedHistorySaverEval)

    end = timeit.default_timer()
    deltaTime = end - start
    deltaTimeMins = deltaTime / 60
    print("Runtime: " + str(deltaTime) + " seconds  (" + str(deltaTimeMins) + " minutes)")

    print("Rate: " + str(args.max_steps / deltaTime) + " steps/second")
    print("SimplificationStr: " + str(args.simplification_str))


def interactive_run(env):
    ob, info = env.reset()
    while True:
        print(clean(ob), 'Reward', reward, 'Done', done, 'Valid', info)
        ob, reward, done, info = env.step(input())


if __name__ == "__main__":
    main()
