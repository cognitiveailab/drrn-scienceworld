import subprocess
import time
import math
import timeit
import torch
import logger
import argparse
from drrn import DRRN_Agent
from vec_env import VecEnv
from jericho.util import clean

from scienceworld_python_api import VirtualEnv, BufferedHistorySaver
import random
from vec_env import resetWithVariation, resetWithVariationTrain, resetWithVariationTest


def configure_logger(log_dir):
    logger.configure(log_dir, format_strs=['log'])
    global tb
    tb = logger.Logger(log_dir, [logger.make_output_format('tensorboard', log_dir),
                                 logger.make_output_format('csv', log_dir),
                                 logger.make_output_format('stdout', log_dir)])
    global log
    log = logger.log


def evaluate(agent, args, env_step_limit, bufferedHistorySaverEval, extraSaveInfo, nb_episodes=10):

    env = reinitializeEnv(args, env_step_limit, threadNum=args.num_envs+10)     # A threadNum (and therefore port) that shouldn't be used by any of the regular training workers

    scoresOut = []
    with torch.no_grad():

        print("evaluate1")
        for ep in range(nb_episodes):
            total_score = 0
            log("Starting evaluation episode {}".format(ep))
            print("evaluate2")
            extraSaveInfo['evalIdx'] = ep
            score = evaluate_episode(agent, env, env_step_limit, args.simplification_str, bufferedHistorySaverEval, extraSaveInfo)
            log("Evaluation episode {} ended with score {}\n\n".format(ep, score))
            total_score += score
            scoresOut.append(total_score)
        avg_score = total_score / nb_episodes
        
        env.shutdown()

        #return avg_score
        return scoresOut, avg_score

    env.shutdown()


def evaluate_episode(agent, env, env_step_limit, simplificationStr, bufferedHistorySaverEval, extraSaveInfo):
    step = 0
    done = False
    numSteps = 0
    #ob, info = env.reset()
    #variationMin = 101            # PJ todo
    #variationMax = 200
    ob, info = resetWithVariationTest(env, simplificationStr)

    state = agent.build_state([ob], [info])[0]
    log('Obs{}: {} Inv: {} Desc: {}'.format(step, clean(ob), clean(info['inv']), clean(info['look'])))
    print("EvaluateEpisode1")
    while not done:
        #print("numSteps: " + str(numSteps))
        #print("EvaluateEpisode2")
        valid_acts = info['valid']
        valid_ids = agent.encode(valid_acts)
        #print("EvaluateEpisode3")
        _, action_idx, action_values = agent.act([state], [valid_ids], sample=False)
        #print("EvaluateEpisode4")
        action_idx = action_idx[0]
        action_values = action_values[0]
        action_str = valid_acts[action_idx]
        log('Action{}: {}, Q-Value {:.2f}'.format(step, action_str, action_values[action_idx].item()))
        #print("EvaluateEpisode5")
        s = ''

        maxToDisplay = 10   # Max Q values to display, to limit the log size
        numDisplayed = 0
        for idx, (act, val) in enumerate(sorted(zip(valid_acts, action_values), key=lambda x: x[1], reverse=True), 1):
            s += "{}){:.2f} {} ".format(idx, val.item(), act)
            numDisplayed += 1
            if (numDisplayed > maxToDisplay):
                break

        log('Q-Values: {}'.format(s))
        #print("EvaluateEpisode6")
        ob, rew, done, info = env.step(action_str)

        #print("EvaluateEpisode7")
        log("Reward{}: {}, Score {}, Done {}".format(step, rew, info['score'], done))
        #print("EvaluateEpisode8")
        step += 1
        log('Obs{}: {} Inv: {} Desc: {}'.format(step, clean(ob), clean(info['inv']), clean(info['look'])))
        state = agent.build_state([ob], [info])[0]
        #print("EvaluateEpisode9")

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

    max_steps = int(math.ceil(max_steps / args.num_envs))
    numEpisodes = 0
    stepsFunctional = 0
    start1 = timeit.default_timer()


    # Reinit environments
    obs, infos = envs.reset()

    states = agent.build_state(obs, infos)
    valid_ids = [agent.encode(info['valid']) for info in infos]
    for step in range(1, max_steps+1):
        stepsFunctional = step * envs.num_envs

        print("-------------------")
        print("Step " + str(step))
        print("")
        end = timeit.default_timer()
        deltaTime = end - start1
        deltaTimeMins = deltaTime / 60
        print("Started at runtime: " + str(deltaTime) + " seconds  (" + str(deltaTimeMins) + " minutes)")
        print("")

        action_ids, action_idxs, _ = agent.act(states, valid_ids)
        action_strs = [info['valid'][idx] for info, idx in zip(infos, action_idxs)]
        #print("StartStep")
        obs, rewards, dones, infos = envs.step(action_strs)
        #print("DoneCheck")
        for done, info in zip(dones, infos):
            if done:
                #tb.logkv_mean('EpisodeScore', info['score'])
                tb.logkv('EpisodeScore', info['score'])
                print("EPISODE SCORE: " + str(info['score']))
                print("EPISODE SCORE: " + str(info['score']) + " STEPS: " + str(stepsFunctional) + " EPISODES: " + str(numEpisodes))

                ## Reset, too?  (note, this only works for single thread?)
                #variationIdx = random.randrange(0, 100)          # train on range 0-20
                #envs.reset()
                #env.resetWithVariation(variationIdx)
                #envs = VecEnv(args.num_envs, env)
                runHistory = info['runHistory']
                #print("Run History (Episode " + str(numEpisodes) + ")")
                #print(runHistory)
                bufferedHistorySaverTrain.storeRunHistory(runHistory, numEpisodes, notes={'step':step})
                bufferedHistorySaverTrain.saveRunHistoriesBufferIfFull(maxPerFile=args.maxHistoriesPerFile)

                numEpisodes += 1

        

        #print("BuildState")
        next_states = agent.build_state(obs, infos)
        #print("Encode")
        next_valids = [agent.encode(info['valid']) for info in infos]
        #print("Observe")
        for state, act, rew, next_state, valids, done in \
            zip(states, action_ids, rewards, next_states, next_valids, dones):
            agent.observe(state, act, rew, next_state, valids, done)
        states = next_states
        valid_ids = next_valids

        if step % log_freq == 0:
            #print("LoggingFreq1")
            tb.logkv('Step', step)
            tb.logkv('StepsFunctional', step*envs.num_envs)
            tb.logkv("FPS", int((step*envs.num_envs)/(time.time()-startTime)))
            tb.logkv('numEpisodes', numEpisodes)
            tb.logkv('taskIdx', args.task_idx)
            tb.logkv('GPU_mem', agent.getMemoryUsage())

            #print("LoggingFreq2")
            print("*************************")
            print("Step:            " + str(step))
            print("StepsFunctional: " + str(step*envs.num_envs))
            print("FPS:             " + str( (step*envs.num_envs)/(time.time()-startTime)) )
            print("numEpisodes:     " + str(numEpisodes))
            print("taskIdx:         " + str(args.task_idx))
            print("GPU_mem:         " + str(agent.getMemoryUsage()))
            print("*************************")
            ### tb.dumpkvs()    # This takes a very long time?
            #print("LoggingFreq3")
            #print("numEpisodes: " + str(numEpisodes))

        if step % update_freq == 0:
            #print("Update0")
            loss = agent.update()
            #print("Update1")
            if loss is not None:
                tb.logkv_mean('Loss', loss)

        if step % checkpoint_freq == 0:
            #print("SaveBefore")
            agent.save("-steps" + str(stepsFunctional) + "-eps" + str(numEpisodes))
            #print("SaveAfter")
        if step & flush_cache_freq == 0:            # PJ: Added to see if this fixes the freezing issue around 1M iterations (where nvidia-smi reports 16GB of GPU memory usage)
            #print("ClearGPUCache")
            agent.clearGPUCache()
        if step % eval_freq == 0:

            # Make an evaluation environment
            #variationIdx = random.randrange(150, 170)          # Evaluate on range 20-40
            #print("Reset1")
            ### envs.reset()            ## PJ: DO NOT RESET anymore, eval env is it's own separate env
            #env.resetWithVariation(variationIdx)
            #envs = VecEnv(args.num_envs, env)

            #print("Reset2")
            extraSaveInfo = {'numEpisodes':numEpisodes, 'numSteps':step, 'stepsFunctional:':stepsFunctional, 'maxHistoriesPerFile':args.maxHistoriesPerFile}
            eval_scores, avg_eval_score = evaluate(agent, args, args.env_step_limit, bufferedHistorySaverEval, extraSaveInfo)

            #print("Reset3")
            tb.logkv('EvalScore', avg_eval_score)
            tb.logkv('numEpisodes', numEpisodes)
            tb.dumpkvs()

            for eval_score in eval_scores:
                print("EVAL EPISODE SCORE: " + str(eval_score))
                print("EVAL EPISODE SCORE: " + str(eval_score) + " STEPS: " + str(stepsFunctional) + " EPISODES: " + str(numEpisodes))

            #print("Reset4")
            # Reset environments to training
            #variationIdx = random.randrange(0, 100)          # train on range 0-20
            envs.reset()
            #env.resetWithVariation(variationIdx)
            #envs = VecEnv(args.num_envs, env)
            #print("Reset5")


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
    parser.add_argument('--var_idx', default=0, type=int)
    parser.add_argument('--maxHistoriesPerFile', default=1000, type=int)
    parser.add_argument('--historySavePrefix', default='saveout', type=str)

    parser.add_argument('--simplification_str', default='', type=str)

    return parser.parse_args()


##PJ
def reinitializeEnv(args, envStepLimit, threadNum):
    jarPath = "virtualenv-scala-assembly-1.0.jar"
    env = VirtualEnv("", jarPath, envStepLimit, threadNum=threadNum)      # unusual threadnum so as not to conflict with other running jobs
    taskNames = env.getTaskNames()
    #taskName = taskNames[0]        # Just get first task
    taskName = taskNames[args.task_idx]        # Just get first task

    maxVariations = env.getMaxVariations(taskName)
    variationIdx = random.randrange(0, maxVariations)           # Pick a random variation
    #variationIdx = args.var_idx    ## Variation IDX currently ignored
    print("NOTE: Generating random variation: " + str(variationIdx))
    env.load(taskName, variationIdx, args.simplification_str)
    initialObs, initialDict = env.reset()

    return env



# def start_redis():
#     print('Starting Redis')
#     subprocess.Popen(['redis-server', '--save', '\"\"', '--appendonly', 'no'])
#     time.sleep(1)

def main():
    ## assert jericho.__version__ == '2.1.0', "This code is designed to be run with Jericho version 2.1.0."
    args = parse_args()
    print(args)
    configure_logger(args.output_dir)
    #start_redis()
    agent = DRRN_Agent(args)

    ## agent.load(path="logs/", suffixStr="-steps576000-eps3473")

    # Try to load agent
    #loadPath = "logs2-findcrash4/"
    #agent.load(loadPath)

    # Jericho initialization
    #env = JerichoEnv(args.rom_path, args.seed, args.env_step_limit)
    #envs = VecEnv(args.num_envs, env)
    #env.create() # Create the environment for evaluation

    # ScienceWorld Initialization?
    #env = reinitializeEnv(args)
    #simplificationStr = "openDoors"
    envs = VecEnv(args.num_envs, args)

    taskIdx = args.task_idx
    bufferedHistorySaverTrain = BufferedHistorySaver(filenameOutPrefix = args.historySavePrefix + "-task" + str(taskIdx) + "-train")
    bufferedHistorySaverEval = BufferedHistorySaver(filenameOutPrefix = args.historySavePrefix + "-task" + str(taskIdx) + "-eval")


    start = timeit.default_timer()

    print("SimplificationStr: " + str(args.simplification_str))

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
