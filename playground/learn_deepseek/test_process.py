import subprocess
import threading

def start_demon_process(command: str, log_file_name: str):
    def _start_demon_process(command: str, log_file_name: str):
        outfile = open(log_file_name, "w")
        process = subprocess.Popen(command, stdout=outfile, stderr=outfile)
        process.wait()
        outfile.close()
    thread = threading.Thread(target=_start_demon_process, args=(command, log_file_name))
    thread.start()

start_demon_process('mooncake_master', 'mooncake_master.log')

while True:
    print('running...')
    time.sleep(10)