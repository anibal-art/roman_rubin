import time
import subprocess
import concurrent.futures
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def execute_command(command):
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def run_command_with_logging(command):
    try:
        process = execute_command(command)

        # Real-time output logging
        for stdout_line in iter(process.stdout.readline, b''):
            logging.info(stdout_line.decode().strip())

        # Wait for the process to finish and capture any remaining output
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            logging.info(f"Command {command} executed successfully")
        else:
            logging.error(f"Command {command} failed with error: {stderr.decode()}")

    except Exception as e:
        logging.error(f"Exception occurred while executing command {command}: {str(e)}")


def run_parallel(path_ephemerides, path_dataslice, path_TRILEGAL_set, path_GENULENS_set, path_to_save_fit, path_to_save_model, model, system_type, algo,
               N_tr):
    commands = [
        ["python", "-c",
         f"from functions_roman_rubin import sim_fit; sim_fit({i},"
         f"'{system_type}','{model}','{algo}', '{path_TRILEGAL_set}','{path_GENULENS_set}', '{path_to_save_model}', '{path_to_save_fit}'"
         f", '{path_ephemerides}', '{path_dataslice}')"]
        for i in range(250000)
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=N_tr) as executor:
        futures = {executor.submit(run_command_with_logging, command): command for command in commands}

        for future in concurrent.futures.as_completed(futures):
            command = futures[future]
            try:
                future.result()
            except Exception as e:
                logging.error(f"Exception occurred for command {command}: {str(e)}")

from pathlib import Path
import re

def run_parallel_read_fit(nset, path_run, path_ephemerides, path_to_save_fit, model, algo, N_tr):
    
    directory = Path(path_run+f"/set_sim{nset}/")
    # Regex pattern
   # print(directory)
    pattern = re.compile(r'Event_(\d+)\.h5$')
    # List to store extracted numbers
    event_numbers = []
    # Loop over all .h5 files
    for file in directory.glob("*.h5"):
        match = pattern.search(file.name)
        if match:
            event_numbers.append(int(match.group(1)))
    #print('hasta aca corre')
    #print('event_numbers',event_numbers) 
   #3print('event_numbers ',event_numbers)
    commands = [
        ["python", "-c",
         f"from functions_roman_rubin import read_fit; read_fit({nsource},"
         f"'{nset}','{path_run}','{model}','{algo}','{path_to_save_fit}', "
         f"'{path_ephemerides}')"]
        for nsource in event_numbers
    ]
    print(commands)
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_tr) as executor:
        futures = {executor.submit(run_command_with_logging, command): command for command in commands}

        for future in concurrent.futures.as_completed(futures):
            command = futures[future]
            try:
                future.result()
            except Exception as e:
                logging.error(f"Exception occurred for command {command}: {str(e)}")
