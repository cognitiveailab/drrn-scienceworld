# batchSubmission.py
import os
import time

# SEEDNUM , TASKID

templateStr = """
version: v2
description: sciworld-may31-drrn-8x100k-taskTASKID-seedSEEDNUM
tasks:
  - name: sciworld-may31-drrn-8x100k-taskTASKID-seedSEEDNUM
    image:
      beaker: peterj/sciworld-drrn2c
    arguments: [python3, train-scienceworld.py, --num_envs=8, --max_steps=100000, --task_idx=TASKID, --simplification_str=easy, --priority_fraction=0.50, --memory_size=100000, --env_step_limit=100, --log_freq=100, --checkpoint_freq=5000, --eval_freq=1000, --seed=SEEDNUM]
    result:
      path: /results1/drrn1/
    resources:
      gpuCount: 1
    context:
      cluster: ai2/raja_p100
      priority: normal
"""
template_command = "python train-scienceworld.py --num_envs=8 --max_steps=100000 --task_idx=TASKID --simplification_str=easy --priority_fraction=0.50 --memory_size=100000 --env_step_limit=100 --log_freq=100 --checkpoint_freq=5000 --eval_freq=1000 --seed=SEEDNUM --output_dir logs/drrn-8x100k-taskTASKID-seedSEEDNUM"


def populateTemplate(taskId, seedNum):
    outStr = template_command
    outStr = outStr.replace("SEEDNUM", str(seedNum))
    outStr = outStr.replace("TASKID", str(taskId))

    return outStr

def writeTemplate(filenameOut, strOut):
    print("Writing " + filenameOut)
    f = open(filenameOut, 'w')
    f.write(strOut)
    f.close

def submitJob(filenameToRun):
    print("Submitting " + filenameToRun)
    runStr = "beaker experiment create " + filenameToRun
    os.system(runStr)
    time.sleep(1)

#
#   Main
#

numJobs = 0
for seed in range(0, 1):
    for taskIdx in range(0, 30):
        tempFilename = "submit.yml"

        #print("Creating job (" + str(numJobs) + "): Task: " + str(taskIdx) + " seed: " + str(seed))
        scriptStr = populateTemplate(taskIdx, seed)
        #writeTemplate(tempFilename, scriptStr)
        print(scriptStr)
        #submitJob(tempFilename)

        #time.sleep(1)
        numJobs += 1
        #print("")
        #print(populateTemplate(10, 2))

print("Submitted " + str(numJobs) + " jobs.")
