#!/bin/bash
#python3 train.py --rom_path ../z-machine-games-master/jericho-game-suite/zork1.z5 --num_envs=1
#python3 train1-sciworld.py --num_envs=1 --max_steps=10000 --task_idx=13 --var_idx=0 --env_step_limit=25 --log_freq=10 --eval_freq=250
#python3 train1-sciworld.py --num_envs=10 --max_steps=10000 --task_idx=13 --var_idx=0 --env_step_limit=100 --log_freq=10 --eval_freq=1000

#python3 train1-sciworld.py --num_envs=10 --max_steps=2000000 --task_idx=13 --var_idx=0 --priority_fraction=0.50 --env_step_limit=200 --log_freq=100 --checkpoint_freq=1000 --eval_freq=1000 |& tee out-findcrash5.txt
#python3 train1-sciworld.py --num_envs=64 --max_steps=5000000 --task_idx=13 --var_idx=0 --priority_fraction=0.50 --memory_size=100000 --env_step_limit=200 --log_freq=100 --checkpoint_freq=1000 --eval_freq=1000 |& tee out-findcrash5.txt

#python3 train1-sciworld.py --num_envs=16 --max_steps=5000000 --task_idx=13 --var_idx=0 --simplification_str=teleportAction,noElectricalAction,openDoors --priority_fraction=0.50 --memory_size=100000 --env_step_limit=100 --log_freq=100 --checkpoint_freq=1000 --eval_freq=1000 |& tee out-findcrash11-load.txt
python3 train1-sciworld.py --num_envs=16 --max_steps=5000000 --task_idx=13 --var_idx=0 --simplification_str=teleportAction,noElectricalAction,openDoors --priority_fraction=0.50 --memory_size=100000 --env_step_limit=100 --log_freq=100 --checkpoint_freq=2000 --eval_freq=1000 |& tee out-findcrash12.txt

