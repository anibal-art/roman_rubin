import time
import subprocess
import time
import os
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



def fitea():
    for i in range(2500):

        print('---------------------------------------------------')
        print('------------ FIT EVENT', i, 'STARTING!! ----------')
        print('---------------------------------------------------')
#         tinit = time()

        try:
            process = subprocess.Popen(["python", "-c", f"from fit_application import fit_light_curve; fit_light_curve('/home/anibal/files_db/august_2023/Event_{i}.txt', 'TRF')"])
            wait_timeout(process, 15*60 )
        except RuntimeError as e:
            print("Process error:", str(e))
            continue

#         tfin = time()
#         print("Event", i, "completed in", tfin - tinit, "seconds.")

fitea()

