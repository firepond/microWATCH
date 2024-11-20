# this runs on host, send task to device and get the result

# protocol specification:
# H\n : hello message, send back hello and board name and programming language

# A\n : ask for available commands, send back available commands

# M+ModelName\n : set model name, send back model name
# M\n : ask for model name, send back model name

# D+DataName\n : set data name, send back data name
# D\n : ask for data name, send back data name

# S\n : start model, sned banck S when ready to start

# F\n : for device to send this when model is done only, not for host to send

# E\n Error, for device to send this when there is an error
# > : device ready to receive command

# with this code from main.py:

from collections import deque
import subprocess
import sys
import threading
import time
from queue import Queue
from usb_logger import power_log

sps = 100
max_size = sps * 1000
data_queue = deque(maxlen=max_size)


# get data from the queue whenever needed
def get_latest_data():
    if data_queue:  # Check if the queue is not empty
        return list(data_queue)  # Return a snapshot of the current data in the queue
    return []


#  use usb_logger.py to log the energy consumption of the target device
#  execute it in another thread and read form it when needed
# Start the producer thread
producer_thread = threading.Thread(target=power_log, args=(data_queue,))
producer_thread.daemon = True
producer_thread.start()


# use keyboard as stdin so the idf.py will be happy
proc = subprocess.Popen(
    "idf.py --port /dev/ttyUSB0 monitor", stdin=None, stdout=subprocess.PIPE, shell=True
)


# Real time stdout of subprocess
stdout = []

# stop the subprocess
proc.terminate()
