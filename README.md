# DRRN Agent (Modified for ScienceWorld)

This repository contains a reference implementation DRRN as mentioned in [Interactive Fiction Games: A Colossal Adventure](https://arxiv.org/abs/1909.05398), that has been modified for use with the [ScienceWorld](https://www.github.com/allenai/ScienceWorld) environment.


# Quickstart

Install Dependencies:
```bash
# Clone repository
git clone https://github.com/cognitiveailab/drrn-scienceworld.git
cd drrn-scienceworld

# Create conda environment
conda create --name drrn-scienceworld python=3.8
conda activate drrn-scienceworld
pip install -r requirements.txt

```

An example of training the DRRN model (using 8 parallel envs, for 10k training steps, evaluating on dev every 1k steps):
```bash
cd drrn
python train-scienceworld.py --num_envs=8 --max_steps=10000 --task_idx=13 --simplification_str=easy --priority_fraction=0.50 --memory_size=100000 --env_step_limit=100 --eval_freq=1000 --eval_set=dev --historySavePrefix=drrn-task13-results-seed0-dev
```
Here:
- **max_steps:** Maximum number of steps to train for (per environment)
- **num_envs:** The number of environments to simultaneously use during training (8 is a common number)
- **task_idx:** The ScienceWorld task index (0-29). *See **task list** below*
- **env_step_limit:** the maximum number of steps to run an environment for, before it times out and resets (100 typical)
- **eval_freq:** the number of steps between evaluations
- **eval_set:** which set to perform the evaluations on (dev or test)
- **historySavePrefix:** the filename prefix for saving the output history files, which contain full logs to calculate final scores, plot performance curves, examing action history, etc.
- **priority_fraction** and **memory_size**: Hyperparameters for the DRRN model (see paper for more information).

This configuration generally takes about 1-2 hours to run (to 10k steps).

## ScienceWorld Task List
```
TASK LIST:
    0: 	                                                 task-1-boil  (30 variations)
    1: 	                        task-1-change-the-state-of-matter-of  (30 variations)
    2: 	                                               task-1-freeze  (30 variations)
    3: 	                                                 task-1-melt  (30 variations)
    4: 	             task-10-measure-melting-point-(known-substance)  (436 variations)
    5: 	           task-10-measure-melting-point-(unknown-substance)  (300 variations)
    6: 	                                     task-10-use-thermometer  (540 variations)
    7: 	                                      task-2-power-component  (20 variations)
    8: 	   task-2-power-component-(renewable-vs-nonrenewable-energy)  (20 variations)
    9: 	                                   task-2a-test-conductivity  (900 variations)
   10: 	             task-2a-test-conductivity-of-unknown-substances  (600 variations)
   11: 	                                          task-3-find-animal  (300 variations)
   12: 	                                    task-3-find-living-thing  (300 variations)
   13: 	                                task-3-find-non-living-thing  (300 variations)
   14: 	                                           task-3-find-plant  (300 variations)
   15: 	                                           task-4-grow-fruit  (126 variations)
   16: 	                                           task-4-grow-plant  (126 variations)
   17: 	                                        task-5-chemistry-mix  (32 variations)
   18: 	                task-5-chemistry-mix-paint-(secondary-color)  (36 variations)
   19: 	                 task-5-chemistry-mix-paint-(tertiary-color)  (36 variations)
   20: 	                             task-6-lifespan-(longest-lived)  (125 variations)
   21: 	         task-6-lifespan-(longest-lived-then-shortest-lived)  (125 variations)
   22: 	                            task-6-lifespan-(shortest-lived)  (125 variations)
   23: 	                               task-7-identify-life-stages-1  (14 variations)
   24: 	                               task-7-identify-life-stages-2  (10 variations)
   25: 	                       task-8-inclined-plane-determine-angle  (168 variations)
   26: 	             task-8-inclined-plane-friction-(named-surfaces)  (1386 variations)
   27: 	           task-8-inclined-plane-friction-(unnamed-surfaces)  (162 variations)
   28: 	                    task-9-mendellian-genetics-(known-plant)  (120 variations)
   29: 	                  task-9-mendellian-genetics-(unknown-plant)  (480 variations)
```

# Hardware requirements
This code generally runs best with at least num_envs+1 CPU cores.

The GPU memory requirements are variable, but generally stay below 8gb.


# Known issues

- *Model saving with manys steps*: Very occassionally, on very long runs (generally 1M+ steps), the periodic pickling the model when saving checkpoints runs into issues and freezes.  The cause is unknown, but as a workaround the save has been wrapped in a timeout, so that if it takes longer than 2 minutes to save the model, the checkpoint is not saved and training continues.  Subsequent checkpoints usually save without issue.


# Citing

If this DRRN agent is helpful in your work, please cite the following:

```
@article{hausknecht19colossal,
  title={Interactive Fiction Games: A Colossal Adventure},
  author={Matthew Hausknecht and Prithviraj Ammanabrolu and Marc-Alexandre C{\^{o}}t{\'{e}} and Xingdi Yuan},
  journal={CoRR},
  year={2019},
  volume={abs/1909.05398}
}
```
