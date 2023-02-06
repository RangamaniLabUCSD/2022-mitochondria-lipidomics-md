from string import Template
from pathlib import Path
import os
import re

### MINIMIZATION
min_template = """
#!/bin/bash
FIRST=`qsub run_equilibration0.pbs`
SECOND=`qsub -W depend=afterok:$FIRST run_equilibration1.pbs`
THIRD=`qsub -W depend=afterok:$SECOND run_equilibration2.pbs`
FOURTH=`qsub -W depend=afterok:$THIRD run_equilibration3.pbs`
FIFTH=`qsub -W depend=afterok:$FOURTH run_equilibration4.pbs`
"""

### RUN 10 jobs ###
# run_template = """
# #!/bin/bash
# FIRST=`qsub run_production1.pbs`
# SECOND=`qsub -W depend=afterok:$FIRST run_production2.pbs`
# THIRD=`qsub -W depend=afterok:$SECOND run_production3.pbs`
# FOURTH=`qsub -W depend=afterok:$THIRD run_production4.pbs`
# FIFTH=`qsub -W depend=afterok:$FOURTH run_production5.pbs`
# # SIXTH=`qsub -W depend=afterok:$FIFTH run_production6.pbs`
# # SEVENTH=`qsub -W depend=afterok:$SIXTH run_production7.pbs`
# # EIGHTH=`qsub -W depend=afterok:$SEVENTH run_production8.pbs`
# # NINTH=`qsub -W depend=afterok:$EIGHTH run_production9.pbs`
# # TENTH=`qsub -W depend=afterok:$NINTH run_production10.pbs`
# ELEVENTH=`qsub -W depend=afterok:$TENTH run_production10+100.pbs`
# """

# runs = [
#     ("minimization1", "step6.2_equilibration.mdp"),
#     ("equilibration0", "step6.2_equilibration.mdp"),
#     ("equilibration1", "step6.3_equilibration.mdp"),
#     ("equilibration2", "step6.4_equilibration.mdp"),
#     ("equilibration3", "step6.5_equilibration.mdp"),
#     ("equilibration4", "step6.6_equilibration.mdp"),
#     ("production1", "step7.2_production.mdp"),
#     ("production2", "step7.2_production.mdp"),
#     ("production3", "step7.2_production.mdp"),
#     ("production4", "step7.2_production.mdp"),
#     ("production5", "step7.2_production.mdp"),
#     ("production6", "step7.2_production.mdp"),
#     ("production7", "step7.2_production.mdp"),
#     ("production8", "step7.2_production.mdp"),
#     ("production9", "step7.2_production.mdp"),
#     ("production10", "step7.2_production.mdp"),
#     ("production10+100", "step7.3_production.mdp"),
# ]


## 5 jobs
run_5_template = """
#!/bin/bash
FIRST=`qsub run_production0.pbs`
SECOND=`qsub -W depend=afterok:$FIRST run_production1.pbs`
THIRD=`qsub -W depend=afterok:$SECOND run_production2.pbs`
FOURTH=`qsub -W depend=afterok:$THIRD run_production3.pbs`
FIFTH=`qsub -W depend=afterok:$FOURTH run_production4.pbs`
"""

runs_5 = [
    ("minimization1", ""),
    ("equilibration0", "step6.2_equilibration.mdp"),
    ("equilibration1", "step6.3_equilibration.mdp"),
    ("equilibration2", "step6.4_equilibration.mdp"),
    ("equilibration3", "step6.5_equilibration.mdp"),
    ("equilibration4", "step6.6_equilibration.mdp"),
    ("production0", "step7.2_production.mdp"),
    ("production1", ""),
    ("production2", ""),
    ("production3", ""),
    ("production4", ""),
]


### 3 production
run_3_template = """
#!/bin/bash
FIRST=`qsub run_production0.pbs`
SECOND=`qsub -W depend=afterok:$FIRST run_production1.pbs`
THIRD=`qsub -W depend=afterok:$SECOND run_production2.pbs`
FOURTH=`qsub -W depend=afterok:$SECOND run_production3.pbs`
FIFTH=`qsub -W depend=afterok:$THIRD run_production+100.pbs`
"""

runs_3 = [
    ("minimization1", ""),
    ("equilibration0", "step6.2_equilibration.mdp"),
    ("equilibration1", "step6.3_equilibration.mdp"),
    ("equilibration2", "step6.4_equilibration.mdp"),
    ("equilibration3", "step6.5_equilibration.mdp"),
    ("equilibration4", "step6.6_equilibration.mdp"),
    ("production0", "step7.2_production.mdp"),
    ("production1", ""),
    ("production2", ""),
    ("production3", ""),
    ("production+100", "step7.3_production.mdp"),
]

JOB_PATTERN = r"""
        (?P<runtype>[a-zA-Z]+)
        (?P<special>\+?)
        (?P<runno>[0-9]+)
    """

base_path = Path("/home/clee2/mito_lipidomics")

sim_path = Path("/oasis/tscc/scratch/clee2/mito_lipidomics")

mdp_path = base_path / "mdps"
script_path = base_path / "scripts"

gmxbin = "/home/clee2/gromacs2022/bin/gmx"
mdpbase = "/home/clee2/mito_lipidomics/mdps_continuation"

queue_base = Path(".")

_regex = re.compile(r"^\s*" + JOB_PATTERN + r"\s*$", re.VERBOSE)

extend_time = 1000000  # picoseconds = 1 us

with Path("./prod_template.pbs").open("r") as fd:
    prod_src = Template(fd.read())

with Path("./min_template.pbs").open("r") as fd:
    min_src = Template(fd.read())

with Path("./equil_template.pbs").open("r") as fd:
    equil_src = Template(fd.read())

with Path("./equil_start_template.pbs").open("r") as fd:
    equil_start_src = Template(fd.read())

with Path("./extend_template.pbs").open("r") as fd:
    extend_src = Template(fd.read())

for system_no in range(1, 25):
    for sys_size in ["", "small"]:
        if sys_size == "small":
            runs = runs_3
            run_template = run_3_template
            system_name = f"{system_no}_small"
        else:
            runs = runs_5
            run_template = run_5_template
            system_name = f"{system_no}"
        print(system_name)

        target_folder = queue_base / system_name

        os.makedirs(target_folder, exist_ok=True)

        initdir = sim_path / f"{system_name}"

        for i in range(1, len(runs)):
            match = _regex.match(runs[i][0]).groupdict()

            d = {
                "SYSTEM_NAME": f"{system_name}_{runs[i][0]}",
                "GMXBIN": gmxbin,
                "INITDIR": initdir,
                "MDP_BASE": mdpbase,
                "PREV_RUN": _regex.match(runs[i - 1][0]).group("runtype"),
            }

            if "equilibration" in match["runtype"]:
                d["CURR_RUN"] = "equilibration"
                d["MDPFILE"] = runs[i][1]
                if int(match["runno"]) == 0:
                    _src = equil_start_src
                else:
                    _src = equil_src
                    d["CHECKPOINT"] = "equilibration"
            elif "minimization" in match["runtype"]:
                _src = min_src
                d["CURR_RUN"] = runs[i][0]
                d["MDPFILE"] = runs[i][1]
            elif "production" in match["runtype"]:
                d["CURR_RUN"] = "production"
                if int(match["runno"]) == 0:
                    _src = prod_src
                    d["CHECKPOINT"] = "equilibration"
                    d["MDPFILE"] = runs[i][1]
                elif match["special"]:
                    _src = prod_src
                    d["CHECKPOINT"] = "production"
                    d["MDPFILE"] = runs[i][1]
                    d["CURR_RUN"] = "production+100"
                else:
                    _src = extend_src
                    d["CHECKPOINT"] = "production"
                    d["EXTEND_TIME"] = extend_time


            # print(i, d)
            result = _src.substitute(d)

            with open(
                target_folder / f"run_{runs[i][0]}.pbs",
                "w",
            ) as fd:
                fd.write(result)

        with open(target_folder / "queue_all.sh", "w") as fd:
            fd.write(run_template)

        with open(target_folder / "queue_minimization.sh", "w") as fd:
            fd.write(min_template)
