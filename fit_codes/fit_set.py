
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

file_events = os.listdir('/home/anibal/files_db/filtered_curves/')
path_ephemerides = '/home/anibal/files_db/james_webb.txt'
path_save = '/home/anibal/roman_rubin/test_badm1/'
path_model = '/home/anibal/files_db/filtered_curves/' #PATH OF THE DIRECTORY OF THE LIGHT CURVE THAT I WANT TO FIT

def fitea():
    for i in range(10):

        print('---------------------------------------------------')
        print('------------ FIT EVENT', file_events[i], 'STARTING!! ----------')
        print('---------------------------------------------------')
#         tinit = time()

        try:
            process = subprocess.Popen([ "python", "-c",
                                         f"from fit_application import fit_light_curve; fit_light_curve('/home/anibal/files_db/filtered_curves/{file_events[i]}',"
                                         f" 'TRF','{path_ephemerides}', '{path_save}')"])
            wait_timeout(process, 15*60 )
        except RuntimeError as e:
            print("Process error:", str(e))
            continue

#         tfin = time()
#         print("Event", i, "completed in", tfin - tinit, "seconds.")

fitea()

