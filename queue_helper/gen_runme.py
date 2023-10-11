


with open("runme_extend100.sh", "w") as fd:
    for i in range(1, 25):
        for small_size in [True]:
            if small_size: 
                fd.write(f"cd {i}_small\n")
            else:
                fd.write(f"cd {i}\n")
            
            fd.write("qsub run_production+100+.pbs\n")
            fd.write("cd ..\n\n")
