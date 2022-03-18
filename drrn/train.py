import os
import time
import torch
import logger
import argparse
import subprocess

import random
import numpy as np
from tqdm import tqdm
from shutil import copyfile
from os.path import join as pjoin

from jericho.util import clean

from env import JerichoEnv
# from vec_env import VecEnv
from vec_env2 import VecEnv
from drrn import DRRN_Agent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_logger(log_dir):
    logger.configure(log_dir, format_strs=['log'])
    global tb
    # tb = logger.Logger(log_dir, [])
    tb = logger.Logger(log_dir, [logger.make_output_format('tensorboard', log_dir),
                                 logger.make_output_format('csv', log_dir),
                                 logger.make_output_format('stdout', log_dir)])
    global log
    log = logger.log


def evaluate_policies(agent, envs, args, max_score):
    log('Evaluating - Argmax and Sampling Policies')

    highscores = np.zeros(args.eval_nb_runs + 1)

    # For reproducibility
    evaluation_seed_rng = np.random.RandomState(args.eval_seed)
    run_seeds = evaluation_seed_rng.randint(1, 65635, size=args.eval_nb_runs)
    run_rngs = [np.random.RandomState(seed) for seed in run_seeds]

    game_name = os.path.basename(str(envs.rom_path))
    # max_score = envs.env.env.get_max_score()

    msg = f"[{os.getpid()}] " + "{}: {mean:5.1f} ± {std:4.1f} [{min:3d}, {median:3d}, {max:3d}] / {max_score:3d} ({norm_avg:4.1f}%) - argmax {argmax:3d}"

    agent.network.eval()
    with torch.no_grad():

        obs, infos = envs.reset()
        with tqdm(total=args.eval_max_steps, leave=False) as pbar:
            for step in range(1, args.eval_max_steps + 1):
                log('Obs{}: {} Inv: {} Desc: {}'.format(step, clean(obs[0]), clean(infos[0]['inv']), clean(infos[0]['look'])))

                states = agent.build_state(obs, infos)
                valid_ids = [agent.encode(info['valid']) for info in infos]

                _, batch_action_idx, batch_action_values = agent.act(states, valid_ids, sample=False)
                batch_act_probs = [torch.softmax(action_values.type(torch.float64), dim=0).cpu() for action_values in batch_action_values]
                batch_action_idx = [int(batch_act_probs[0].argmax())] + [rng.multinomial(1, act_probs).argmax() for rng, act_probs in zip(run_rngs, batch_act_probs[1:])]
                batch_action_str = [info['valid'][idx] for info, idx in zip(infos, batch_action_idx)]

                log('Action{}: {}, Q-Value {:.2f}'.format(step, batch_action_str[0], batch_action_values[0][batch_action_idx[0]]))
                s = ''
                for idx, (act, val) in enumerate(sorted(zip(infos[0]['valid'], batch_action_values[0]), key=lambda x: x[1], reverse=True), 1):
                    s += "{}){:.2f} {} ".format(idx, val.item(), act)
                log('Q-Values: {}'.format(s))

                obs, rews, dones, infos = envs.step(batch_action_str)
                scores = [info["score"] for info in infos]
                highscores = np.maximum(highscores, scores)

                log("Reward{}: {}, Score {}, Done {}".format(step, rews[0], infos[0]['score'], dones[0]))

                if dones[0]:
                    # Replay the game in the hope of achieving a better score.
                    log("Environment reset.")

                argmax_highscore = int(highscores[0])
                sampling_highscores = highscores[1:]
                norm_avg = 100.0 * np.mean(sampling_highscores) / max_score
                status = msg.format(game_name, mean=np.mean(sampling_highscores), std=np.std(sampling_highscores), min=int(np.min(sampling_highscores)), median=int(np.median(sampling_highscores)), max=int(np.max(sampling_highscores)), max_score=max_score, norm_avg=norm_avg, argmax=argmax_highscore)
                pbar.set_description(status)
                pbar.update(1)

    agent.network.train()
    return highscores


def train(agent, eval_env, eval_envs, envs, max_steps, update_freq, eval_freq, checkpoint_freq, log_freq, args, checkpoint):
    start = time.time()

    total_cache_hits = 0
    total_actions = 0
    max_score = eval_env.env.get_max_score()

    best_eval_score_argmax = checkpoint.get("best", {}).get("eval_score_argmax", -np.inf)
    best_mean_eval_scores_sampling = checkpoint.get("best", {}).get("mean_eval_scores_sampling", -np.inf)
    best_eval_scores_sampling = checkpoint.get("best", {}).get("eval_scores_sampling", [-np.inf]*args.eval_nb_runs)
    best_argmax_policy_step = checkpoint.get("best", {}).get("argmax_policy_step", 0)
    best_sampling_policy_step = checkpoint.get("best", {}).get("sampling_policy_step", 0)

    obs, infos = envs.reset()
    if 'env' in checkpoint:  # Reload envs
        envs.set_state(checkpoint['env']['jericho'])
        obs = checkpoint['env']['obs']
        infos = checkpoint['env']['infos']

    states = agent.build_state(obs, infos)
    valid_ids = [agent.encode(info['valid']) for info in infos]

    if checkpoint.get("step", 0) == 0:
        # Evaluate agent at step 0 -> Random policy.
        # Evaluate both argmax and sampling policies.
        eval_scores = evaluate_policies(agent, eval_envs, args, max_score)
        eval_score_argmax = eval_scores[0]
        eval_scores_sampling = eval_scores[1:]

        # Argmax evaluation.
        tb.logkv('step', 0)
        tb.logkv('Argmax - Score', eval_score_argmax)
        tb.logkv('Argmax - Score (%)', eval_score_argmax / max_score)

        # Sampling evaluation.
        tb.logkv('Sampling - Scores', eval_scores_sampling)
        tb.logkv('Sampling - Score - mean (%)', np.mean(eval_scores_sampling) / max_score)
        tb.logkv('Sampling - Score - mean', np.mean(eval_scores_sampling))
        tb.logkv('Sampling - Score - std', np.std(eval_scores_sampling))
        tb.logkv('Sampling - Score - min', np.min(eval_scores_sampling))
        tb.logkv('Sampling - Score - mid', np.median(eval_scores_sampling))
        tb.logkv('Sampling - Score - max', np.max(eval_scores_sampling))

        tb.dumpkvs()

        best_eval_score_argmax = eval_score_argmax
        best_argmax_policy_step = 0
        print(f"[{os.getpid()}] * Random argmax policy has a score of {int(best_eval_score_argmax)}.")

        best_eval_scores_sampling = eval_scores_sampling
        best_mean_eval_scores_sampling = np.mean(eval_scores_sampling)
        best_sampling_policy_step = 0
        print(f"[{os.getpid()}] * Random sampling policy has an avg. score of {np.mean(eval_scores_sampling):5.1f} ± {np.std(eval_scores_sampling):4.1f}.")

    for step in range(checkpoint.get("step", 0) + 1, max_steps+1):
        action_ids, action_idxs, _ = agent.act(states, valid_ids)
        action_strs = [info['valid'][idx] for info, idx in zip(infos, action_idxs)]

        # print(obs[0])
        # print(action_strs[0])
        # print("-=-=-=-=-")

        obs, rewards, dones, infos = envs.step(action_strs)
        for i, (done, info) in enumerate(zip(dones, infos)):
            total_actions += 1
            total_cache_hits += info["cache_hit"]

            if done:
                tb.logkv_mean('EpisodeScore', info['score'])

        next_states = agent.build_state(obs, infos)
        next_valids_ids = [agent.encode(info['valid']) for info in infos]
        for state, act, rew, next_state, valids, done in \
            zip(states, action_ids, rewards, next_states, next_valids_ids, dones):
            agent.observe(state, act, rew, next_state, valids, done)

        states, valid_ids = next_states, next_valids_ids

        trigger_checkpoint = False
        found_better_argmax_policy = False
        found_better_sampling_policy = False

        if step % update_freq == 0:
            loss = agent.update()
            if loss is not None:
                tb.logkv_mean('Loss', loss)

        if step % log_freq == 0:
            print("Time: {:.2f} secs".format(time.time()-start))
            tb.logkv('step', step)
            tb.logkv("FPS", int((step*envs.num_envs)/(time.time()-start)))
            tb.logkv("CacheHitsRatio", np.round(100*total_cache_hits/total_actions, 2))
            tb.logkv("BestArgmaxPolicyStep", best_argmax_policy_step)
            tb.logkv("BestSamplingPolicyStep", best_sampling_policy_step)
            tb.dumpkvs()

        if step % eval_freq == 0:
            # Evaluate both argmax and sampling policies.
            eval_scores = evaluate_policies(agent, eval_envs, args, max_score)
            eval_score_argmax = eval_scores[0]
            eval_scores_sampling = eval_scores[1:]

            # Argmax evaluation.
            tb.logkv('step', step)
            tb.logkv('Argmax - Score', eval_score_argmax)
            tb.logkv('Argmax - Score (%)', eval_score_argmax / max_score)

            # Sampling evaluation.
            tb.logkv('Sampling - Scores', eval_scores_sampling)
            tb.logkv('Sampling - Score - mean (%)', np.mean(eval_scores_sampling) / max_score)
            tb.logkv('Sampling - Score - mean', np.mean(eval_scores_sampling))
            tb.logkv('Sampling - Score - std', np.std(eval_scores_sampling))
            tb.logkv('Sampling - Score - min', np.min(eval_scores_sampling))
            tb.logkv('Sampling - Score - mid', np.median(eval_scores_sampling))
            tb.logkv('Sampling - Score - max', np.max(eval_scores_sampling))

            tb.dumpkvs()

            if eval_score_argmax > best_eval_score_argmax:
                best_eval_score_argmax = eval_score_argmax
                trigger_checkpoint = True
                found_better_argmax_policy = True
                best_argmax_policy_step = step
                print(f"[{os.getpid()}] * Found better argmax policy at step {step} with score {int(best_eval_score_argmax)}.")

            if np.mean(eval_scores_sampling) > best_mean_eval_scores_sampling:
                best_eval_scores_sampling = eval_scores_sampling
                best_mean_eval_scores_sampling = np.mean(eval_scores_sampling)
                trigger_checkpoint = True
                found_better_sampling_policy = True
                best_sampling_policy_step = step
                print(f"[{os.getpid()}] * Found better sampling policy at step {step} with avg. score {np.mean(eval_scores_sampling):5.1f} ± {np.std(eval_scores_sampling):4.1f}.")

        if trigger_checkpoint or step % checkpoint_freq == 0:
            torch.save({
                'step': step,
                'best': {
                    'eval_score_argmax': best_eval_score_argmax,
                    'mean_eval_scores_sampling': best_mean_eval_scores_sampling,
                    'eval_scores_sampling': best_eval_scores_sampling,
                    'argmax_policy_step': best_argmax_policy_step,
                    'sampling_policy_step': best_sampling_policy_step,
                },
                'env': {
                    'jericho': envs.get_state(),
                    'obs': obs,
                    'infos': infos,
                },
                'rng': {
                    'torch': torch.random.get_rng_state(),
                    'cuda_devices': torch.cuda.random.get_rng_state_all(),
                },
                'memory': agent.memory,
                'network': agent.network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                },
                f=args.checkpoint_path)

            if found_better_argmax_policy:
                copyfile(args.checkpoint_path, pjoin(args.output_dir, "best_argmax.ckpt"))

            if found_better_sampling_policy:
                copyfile(args.checkpoint_path, pjoin(args.output_dir, "best_sampling.ckpt"))


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
    parser.add_argument('--checkpoint_path')
    parser.add_argument('--redis_dir')
    parser.add_argument('--eval_freq', default=1000, type=int)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--memory_size', default=500000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gamma', default=.9, type=float)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--clip', default=5, type=float)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--eval_max_steps', default=100, type=int)
    parser.add_argument('--eval_nb_runs', default=16, type=int)
    parser.add_argument('--eval_seed', default=20210821, type=int)
    parser.add_argument('--job_name', default="drrn")
    return parser.parse_args()

from contextlib import contextmanager

@contextmanager
def start_redis(output_dir):
    print(f"Starting Redis (logging to {pjoin(output_dir, 'redis.log')})")

    os.makedirs(output_dir, exist_ok=True)
    redis_process = subprocess.Popen(['redis-server', 'redis.conf', '--dir', output_dir, '--logfile', 'redis.log'],
                                     start_new_session=True)
    time.sleep(1)

    try:
        yield redis_process
    finally:
        redis_process.terminate()
        redis_process.wait()


def main():
    args = parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    with start_redis(args.redis_dir or args.output_dir):  # Database shared across different seeds.

        args.output_dir = pjoin(args.output_dir, f"{args.seed}")
        configure_logger(args.output_dir)

        # Set some paths for checkpointing.
        args.checkpoint_path = args.checkpoint_path or args.output_dir
        os.makedirs(args.checkpoint_path, exist_ok=True)
        args.checkpoint_path = pjoin(args.checkpoint_path, args.job_name + ".ckpt" )

        print(f"[{os.getpid()}] Agent running on device:", device)
        agent = DRRN_Agent(args)

        try:
            # agent.load()  # Resume training
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            agent.memory = checkpoint["memory"]
            agent.network.load_state_dict(checkpoint["network"])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            agent.network = agent.network.to(device)

            torch.random.set_rng_state(checkpoint["rng"]["torch"])
            torch.cuda.random.set_rng_state_all(checkpoint["rng"]["cuda_devices"])
            print(f"[{os.getpid()}] Resume training from step {checkpoint['step']}... (found {args.checkpoint_path})")
        except (FileNotFoundError, TypeError):
            print(f"[{os.getpid()}] New training...")
            checkpoint = {}

        env = JerichoEnv(args.rom_path, args.env_step_limit)
        envs = VecEnv(args.num_envs, args.rom_path, args.env_step_limit)
        eval_envs = VecEnv(args.eval_nb_runs + 1, args.rom_path, args.env_step_limit)  # eval_nb_runs for sampling policy + 1 for argmax policy

        #envs = VecEnv(args.num_envs, env)
        #eval_envs = VecEnv(args.eval_nb_runs + 1, env)  # eval_nb_runs for sampling policy + 1 for argmax policy
        env.create() # Create the environment for evaluation

        train(agent, env, eval_envs, envs, args.max_steps, args.update_freq, args.eval_freq,
            args.checkpoint_freq, args.log_freq, args, checkpoint)


if __name__ == "__main__":
    main()
