


with open("runme_min.sh", "w") as fd:
    for i in range(1, 25):
        for small_size in [True, False]:
            if small_size: 
                fd.write(f"cd {i}_small\n")
            else:
                fd.write(f"cd {i}\n")
            
            fd.write("sh queue_mininimization.sh\n")
            fd.write("cd ..\n\n")
