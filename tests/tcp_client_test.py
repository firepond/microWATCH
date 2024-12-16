import copy
import os
import socket
from collections import deque
import threading
import time
from time import sleep

import numpy as np
import usb_logger


def get_latest_data(data_queue):
    if data_queue:  # Check if the queue is not empty
        return list(data_queue)  # Return a snapshot of the current data in the queue
    return []


def test_model(file_name, data_queue, model_name):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("192.168.50.4", 5000))  # Replace with Raspberry Pi's IP address

    commands = [
        "M+" + model_name,
        "D+datasets/csv/" + file_name,
    ]

    for command in commands:
        client.sendall(command.encode())
        sleep(1)
        response = client.recv(4096).decode()
        print("Server:", response)

    # send "S", when get a "S" response from the server, start the power consumption measurement
    client.sendall(b"S")
    # power_data = get_latest_data(data_queue)

    start_data = get_latest_data(data_queue)[-1]

    response = client.recv(4096).decode()

    print("Server:", response)

    # print the response until get a "F" from the server

    while "F:" not in response:
        response = client.recv(4096).decode()
        print(f"Server:{response} end")
    # prase run times: int after "F:"
    messages = response.split("\n")

    run_count = int(messages[0].split("F:")[-1])
    # parse time:

    time_used = messages[1].split("Time:")[-1].split("ms")[0]
    print(f"Time used: {time_used} ms")

    power_data = get_latest_data(data_queue)
    end_data = power_data[-1]

    start_stamp = start_data[0]
    end_stamp = end_data[0]
    filtered_data = [
        data for data in power_data if data[0] >= start_stamp and data[0] <= end_stamp
    ]
    energy_consumed = (filtered_data[-1][3] - filtered_data[0][3]) / run_count
    print(f"Energy consumed: {energy_consumed} J")

    sleep(1)
    # send "Q" to close the connection
    client.sendall(b"Q")

    client.close()
    return energy_consumed, time_used


def main():
    # start usb_logger in another thread
    sps = 100
    max_size = sps * 100000
    data_queue = deque(maxlen=max_size)
    stop = [False]
    usb_logger_thread = threading.Thread(
        target=usb_logger.power_log, args=(data_queue, stop)
    )
    usb_logger_thread.daemon = True
    usb_logger_thread.start()
    sleep(5)

    result_file_name = "pi_rbocpdms_results.csv"

    result_file = open(result_file_name, "a")
    result_file.write("file_name,  energy_consumed(J), time_used(ms)\n")

    model_name = "RBOCPDMS"
    # uni_variate = True
    for file_name in os.listdir("datasets/csv"):

        print(f"Testing file {file_name}")
        energy_conusmed, time_used = test_model(file_name, data_queue, model_name)
        result_file.write(f"{file_name}, {energy_conusmed}, {time_used}\n")
        result_file.flush()

    result_file.close()
    stop[0] = True
    # close the usb_logger thread
    print("Closing usb_logger thread, join")
    usb_logger_thread.join()
    print("Closed")


if __name__ == "__main__":
    main()
