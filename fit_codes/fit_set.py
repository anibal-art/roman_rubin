import time
def wait_timeout(proc, seconds):
    """Wait for a process to finish, or raise exception after timeout"""
    start = time.time()
    end = start + seconds
    interval = min(seconds / 1000.0, .25)

    while True:
        result = proc.poll()
        if result is not None:
            return result
        if time.time() >= end:
            proc.kill()
            raise RuntimeError("Process timed out")

        time.sleep(interval)

import subprocess
import time
import os
filenames = [f for f in os.listdir('/home/anibalvarela/curvas/') if 'txt' in f]

n = []
for i in range(len(filenames)):
    n.append(filenames[i][6:filenames[i].index('.txt')])

events_list = list(map(int, n))

def fitea():
    takes_to_long = [53, 210, 228, 229, 248, 262, 321, 443]
    for i in events_list:
        if i in takes_to_long:
            print("Event", i, "takes too long. Skipping.")
            continue

        print('---------------------------------------------------')
        print('------------ FIT EVENT', i, 'STARTING!! ----------')
        print('---------------------------------------------------')
#         tinit = time()

        try:
            process = subprocess.Popen(["python", "-c", f"from fit_functions import fit_light_curve; fit_light_curve('/home/anibalvarela/curvas/Event_{i}.txt', 'TRF')"])
            wait_timeout(process, 15*60 )
        except RuntimeError as e:
            print("Process error:", str(e))
            continue

#         tfin = time()
#         print("Event", i, "completed in", tfin - tinit, "seconds.")

fitea()
