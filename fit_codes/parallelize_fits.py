import time
import subprocess
import os
def execute_command(command):
    process = subprocess.Popen(command)
    return process

def main(path_model, path_ephemerides, path_save):
    file_events = os.listdir(path_model)
    commands = []

    for i in range(0, len(file_events)):
        commands.append(["python", "-c",
                         f"from fit_application import fit_light_curve; fit_light_curve('{path_model + file_events[i]}',"
                         f"'TRF','{path_ephemerides}', '{path_save}')"])

    running_processes = []
    max_concurrent_processes = 30  # You can adjust this value
    len_init = max_concurrent_processes

    for command in commands[:max_concurrent_processes]:
        process = execute_command(command)
        running_processes.append(process)

    new_commands = []
    while not len_init == len(commands):
        for process in running_processes:
            return_code = process.poll()
            if return_code is not None:  # Process finished
                time.sleep(10)
                running_processes.remove(process)
                time.sleep(10)
                new_command = commands[len_init]
                new_commands.append(commands[len_init])
                len_init = len_init + 1

                if new_command:
                    time.sleep(10)
                    new_process = execute_command(new_command)
                    running_processes.append(new_process)
                break

        time.sleep(1)  # Adjust the sleep duration as needed