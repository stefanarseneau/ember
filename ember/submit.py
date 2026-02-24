# src/submit.py
from __future__ import annotations
import os
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional

def _qsub_directives(
    job_name: str,
    logs_dir: Path,
    numtasks: int,
    queue: Optional[str],
    project: Optional[str],
    pe: Optional[str],
    slots: int,
    mem: Optional[str],          # e.g. "8G"
    hours: Optional[int],        # walltime in hours
    email: Optional[str],
    hold_jid: Optional[str],
) -> List[str]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "#$ -cwd",
        "#$ -V",
        f"#$ -N {job_name}",
        f"#$ -o {logs_dir}/$JOB_NAME.$JOB_ID.$TASK_ID.out",
        f"#$ -e {logs_dir}/$JOB_NAME.$JOB_ID.$TASK_ID.err",
    ]
    print(f"numtasks : {numtasks}")
    if numtasks and numtasks > 1:
        lines.append(f"#$ -t 1-{numtasks}")
    if queue:
        lines.append(f"#$ -q {queue}")
    if project:
        lines.append(f"#$ -P {project}")
    if pe and slots > 0:
        lines.append(f"#$ -pe {pe} {slots}")
    if mem:
        lines.append(f"#$ -l h_vmem={mem}")
    if hours:
        lines.append(f"#$ -l h_rt={hours:02d}:00:00")
    if email:
        lines += [f"#$ -M {email}", "#$ -m abe"]  # mail at abort, begin, end
    if hold_jid:
        lines.append(f"#$ -hold_jid {hold_jid}")
    return lines

def write_qsub_script(
    script_path: Path,
    ember_cmd: List[str],
    job_name: str = "ember",
    logs_dir: Path = Path("./logs"),
    numtasks: int = 1,
    queue: Optional[str] = None,
    project: Optional[str] = None,
    pe: Optional[str] = None,
    slots: int = 1,
    mem: Optional[str] = None,
    hours: Optional[int] = None,
    email: Optional[str] = None,
    hold_jid: Optional[str] = None,
    conda_env: Optional[str] = None,
    module: Optional[str] = "miniconda",
    extra_exports: Optional[dict[str, str]] = None,
) -> Path:
    directives = _qsub_directives(job_name, logs_dir, numtasks, queue, project, pe, slots, mem, hours, email, hold_jid)
    quoted_cmd = " ".join(shlex.quote(x) for x in ember_cmd)
    exports = ""
    if extra_exports:
        exports = "\n".join(f"export {k}={shlex.quote(v)}" for k, v in extra_exports.items())

    prologue = f"""#!/bin/bash -l
{os.linesep.join(directives)}

set -euo pipefail
module load {module}
{'source ~/.bashrc && conda activate ' + shlex.quote(conda_env) if conda_env else ''}

echo "Host: $(hostname)"
echo "Job:  $JOB_NAME  ID: $JOB_ID  Task: $SGE_TASK_ID"
echo "Cmd:  {quoted_cmd}"
"""

    body = f"""
# Run the command
{quoted_cmd}
"""

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(prologue + body)
    return script_path

def qsub_submit(script_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["qsub", str(script_path)], capture_output=True, text=True, check=False)
