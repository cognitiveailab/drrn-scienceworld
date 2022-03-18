# Template-DQN and DRRN (Modified for ScienceWorld)

This repository contains reference implementations of TDQN and DRRN as mentioned in [Interactive Fiction Games: A Colossal Adventure](https://arxiv.org/abs/1909.05398).

Changelog:
- Marc: Updates for various improvements/newer library versions
- Peter: Hackish, single-thread DRRN modification to run with ScienceWorld

Current issues
- Threads: ScienceWorld API only partially integrated: currently doesn't run more than one thread (if you try this, it may crash)
- ScienceWorld: Currently (Dec 16th/2021) a bug that prevents 'open door' actions from being in the list of valid actions in the Jericho API, so agents won't be able to successfully complete most tasks.  This should be fixed shortly.
- ScienceWorld (Electrical conductivity actions): Dec 15th/2021: These were disabled to reduce the action space and increase simulation speed by about 10x.  Should be re-enabled after completed (or, increase the speed of the generateValidActions() call in the API, which is a major bottleneck right now).
- TDQN: This model hasn't been modified yet.

# Quickstart

Install Dependencies:
```bash
# Clone repository
git clone https://github.com/cognitiveailab/tdqn-scienceworld.git
cd tdqn-scienceworld

# Create conda environment
conda create --name drrn2 python=3.8
conda activate drrn2
pip install -r requirements.txt

```

An example of training the DRRN model:
```bash
cd drrn
python3 train1-sciworld.py --num_envs=1 --max_steps=1000 --task_idx=13 --var_idx=0 --env_step_limit=25 --log_freq=10 --eval_freq=250
```
Here:
- **max_steps:** Maximum number of steps to train for
- **task_idx:** The ScienceWorld task index (0-29). *See **task list** below*
- **var_idx:** The ScienceWorld task variation number
- **env_step_limit:** the maximum number of steps to run an environment for, before it times out and resets
- **eval_freq:** the number of steps between evaluations (evaluations are capped at ~200 steps -- this is currently hardcoded, you can change this in the code if needed).

## ScienceWorld Task List
```
TASK LIST:
    0: 	                                                 task-1-boil  (30 variations)
    1: 	                        task-1-change-the-state-of-matter-of  (30 variations)
    2: 	                                               task-1-freeze  (30 variations)
    3: 	                                                 task-1-melt  (30 variations)
    4: 	             task-10-measure-melting-point-(known-substance)  (875 variations)
    5: 	           task-10-measure-melting-point-(unknown-substance)  (12500 variations)
    6: 	                                     task-10-use-thermometer  (27000 variations)
    7: 	                                      task-2-power-component  (20 variations)
    8: 	   task-2-power-component-(renewable-vs-nonrenewable-energy)  (20 variations)
    9: 	                                   task-2a-test-conductivity  (2000 variations)
   10: 	             task-2a-test-conductivity-of-unknown-substances  (2000 variations)
   11: 	                                          task-3-find-animal  (300 variations)
   12: 	                                    task-3-find-living-thing  (300 variations)
   13: 	                                task-3-find-non-living-thing  (300 variations)
   14: 	                                           task-3-find-plant  (300 variations)
   15: 	                                           task-4-grow-fruit  (126 variations)
   16: 	                                           task-4-grow-plant  (126 variations)
   17: 	                                        task-5-chemistry-mix  (4 variations)
   18: 	                task-5-chemistry-mix-paint-(secondary-color)  (12 variations)
   19: 	                 task-5-chemistry-mix-paint-(tertiary-color)  (12 variations)
   20: 	                             task-6-lifespan-(longest-lived)  (125 variations)
   21: 	         task-6-lifespan-(longest-lived-then-shortest-lived)  (125 variations)
   22: 	                            task-6-lifespan-(shortest-lived)  (125 variations)
   23: 	                               task-7-identify-life-stages-1  (14 variations)
   24: 	                               task-7-identify-life-stages-2  (14 variations)
   25: 	                       task-8-inclined-plane-determine-angle  (168 variations)
   26: 	             task-8-inclined-plane-friction-(named-surfaces)  (1386 variations)
   27: 	           task-8-inclined-plane-friction-(unnamed-surfaces)  (210 variations)
   28: 	                    task-9-mendellian-genetics-(known-plant)  (120 variations)
   29: 	                  task-9-mendellian-genetics-(unknown-plant)  (480 variations)
```


# Old instructions
Train TDQN:
```bash
cd tdqn/tdqn && python3 train.py --rom_path <path_to_your_rom_file>
```

Train DRRN:
```bash
cd tdqn/drrn && python3 train.py --rom_path <path_to_your_rom_file>
```

# Citing

If these agents are helpful in your work, please cite the following:

```
@article{hausknecht19colossal,
  title={Interactive Fiction Games: A Colossal Adventure},
  author={Matthew Hausknecht and Prithviraj Ammanabrolu and Marc-Alexandre C{\^{o}}t{\'{e}} and Xingdi Yuan},
  journal={CoRR},
  year={2019},
  volume={abs/1909.05398}
}
```
