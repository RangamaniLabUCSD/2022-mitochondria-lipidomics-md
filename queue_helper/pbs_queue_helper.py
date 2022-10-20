from string import Template
from pathlib import Path
import os

min_template = """
#!/bin/bash
FIRST=`qsub run_equilibration0.pbs`
SECOND=`qsub -W depend=afterok:$FIRST run_equilibration1.pbs`
THIRD=`qsub -W depend=afterok:$SECOND run_equilibration2.pbs`
FOURTH=`qsub -W depend=afterok:$THIRD run_equilibration3.pbs`
FIFTH=`qsub -W depend=afterok:$FOURTH run_equilibration4.pbs`
"""

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

run_template = """
#!/bin/bash
FIRST=`qsub run_production1.pbs`
SECOND=`qsub -W depend=afterok:$FIRST run_production2.pbs`
THIRD=`qsub -W depend=afterok:$SECOND run_production3.pbs`
FOURTH=`qsub -W depend=afterok:$THIRD run_production4.pbs`
FIFTH=`qsub -W depend=afterok:$FOURTH run_production5.pbs`
SIXTH=`qsub -W depend=afterok:$FIFTH run_production5+100.pbs`
"""

runs = [
    ("minimization1", "step6.2_equilibration.mdp"),
    ("equilibration0", "step6.2_equilibration.mdp"),
    ("equilibration1", "step6.3_equilibration.mdp"),
    ("equilibration2", "step6.4_equilibration.mdp"),
    ("equilibration3", "step6.5_equilibration.mdp"),
    ("equilibration4", "step6.6_equilibration.mdp"),
    ("production1", "step7.2_production.mdp"),
    ("production2", "step7.2_production.mdp"),
    ("production3", "step7.2_production.mdp"),
    ("production4", "step7.2_production.mdp"),
    ("production5", "step7.2_production.mdp"),
    ("production5+100", "step7.3_production.mdp"),
]


base_path = Path("/home/clee2/mito_lipidomics")

sim_path = Path("/oasis/tscc/scratch/clee2/mito_lipidomics")

mdp_path = base_path / "mdps"
script_path = base_path / "scripts"

gmxbin = "/home/clee2/gromacs2022/bin/gmx"
mdpbase = "/home/clee2/mito_lipidomics/mdps"

queue_base = Path(".")

for i in range(8,12):
    template_file = Path("./template.pbs")
    with template_file.open("r") as fd:
        src = Template(fd.read())

    min_template_file = Path("./min_template.pbs")
    with min_template_file.open("r") as fd:
        min_src = Template(fd.read())

    system_name = f"{i}"

    target_folder = queue_base / system_name

    os.makedirs(target_folder, exist_ok=True)

    initdir = sim_path / f"{i}"

    for i in range(1,len(runs)):
        if 'equilibration' in runs[i][0]:
            _src = min_src
        else:
            _src = src
        d = {
            "SYSTEM_NAME": f"{system_name}_{runs[i][0]}",
            "GMXBIN": gmxbin,
            "INITDIR": initdir,
            "MDP_BASE": mdpbase,
            "MDPFILE" : runs[i][1],
            "PREV_RUN": runs[i-1][0],
            "CURR_RUN": runs[i][0],
        }

        result = _src.substitute(d)

        with open(
            target_folder
            / f"run_{runs[i][0]}.pbs",
            "w",
        ) as fd:
            fd.write(result)

    with open(target_folder / "queue_all.sh", 'w') as fd:
        fd.write(run_template)

    with open(target_folder / "queue_minimization.sh", 'w') as fd:
        fd.write(min_template)


# template_file = Path("./template.pbs")
# with template_file.open("r") as fd:
#     src = Template(fd.read())

# min_template_file = Path("./min_template.pbs")
# with min_template_file.open("r") as fd:
#     min_src = Template(fd.read())

# system_name = f"0enth_na"

# target_folder = queue_base / system_name

# os.makedirs(target_folder, exist_ok=True)

# initdir = sim_path / f"{system_name}"

# for i in range(1,len(runs)):
#     if 'equilibration' in runs[i][0]:
#         _src = min_src
#     else:
#         _src = src
#     d = {
#         "SYSTEM_NAME": f"{system_name}_{runs[i][0]}",
#         "GMXBIN": gmxbin,
#         "INITDIR": initdir,
#         "MDP_BASE": mdpbase,
#         "MDPFILE" : runs[i][1],
#         "PREV_RUN": runs[i-1][0],
#         "CURR_RUN": runs[i][0],
#     }

#     result = _src.substitute(d)
#     print(f"run_{runs[i][0]}")
#     with open(
#         target_folder
#         / f"run_{runs[i][0]}.pbs",
#         "w",
#     ) as fd:
#         fd.write(result)

# with open(target_folder / "queue_all.sh", 'w') as fd:
#     fd.write(run_template)

# with open(target_folder / "queue_minimization.sh", 'w') as fd:
#     fd.write(min_template)
