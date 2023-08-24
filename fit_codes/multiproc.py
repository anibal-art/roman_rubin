from subprocess import Popen
import os
file_events = os.listdir('/home/anibal/files_db/filtered_curves/')
path_ephemerides = '/home/anibal/files_db/james_webb.txt'
path_save = '/home/anibal/roman_rubin/test_badm1/'
path_model = '/home/anibal/files_db/filtered_curves/' #PATH OF THE DIRECTORY OF THE LIGHT CURVE THAT I WANT TO FIT

for j in range(0,1000,10):
    commands = []
    for i in range(j,j+10):
        commands.append(["python", "-c",
         f"from fit_application import fit_light_curve; fit_light_curve('/home/anibal/files_db/filtered_curves/{file_events[i]}',"
         f" 'TRF','{path_ephemerides}', '{path_save}')"])

    processes = [Popen(cmd) for cmd in commands]
    for process in processes:
        process.wait()