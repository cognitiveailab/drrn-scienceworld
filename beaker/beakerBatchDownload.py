import subprocess
import argparse
import shlex
import json

parser = argparse.ArgumentParser(
    description="A command for downloading results for all experiments in a workspace")
parser.add_argument("--text", help="text to filter the returned experiments by.", required=True)
parser.add_argument("--workspace", help="the target workspace")

args = parser.parse_args()

experiments = json.loads(subprocess.check_output(shlex.split(f"""
    beaker workspace experiments {args.workspace} \
        --text '{args.text}'
        --format json
""")))

for exp in experiments:
    for exec in exp.get("executions", []):
        dataset_id = exec.get("result", {}).get("beaker")
        if dataset_id is not None:
            print(f"downloading dataset {dataset_id}, results of {exp.get('id')}...")
            subprocess.check_call(shlex.split(f"""
                beaker dataset fetch {dataset_id} --output {dataset_id}
            """))