import time
import subprocess
import concurrent.futures
import logging
import numpy as np
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


def parallel_test_width(N_tr):
    commands = [
        ["python", "-c",
         f"from test_width import fit_test; fit_test({rango})"]
        for rango in np.logspace(-4,1,30)
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=N_tr) as executor:
        futures = {executor.submit(run_command_with_logging, command): command for command in commands}

        for future in concurrent.futures.as_completed(futures):
            command = futures[future]
            try:
                future.result()
            except Exception as e:
                logging.error(f"Exception occurred for command {command}: {str(e)}")