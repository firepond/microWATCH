import copy
import os
import socket
from collections import deque
import threading
import time
from time import sleep
import usb_logger


def get_latest_data(data_queue):
    if data_queue:  # Check if the queue is not empty
        return list(data_queue)  # Return a snapshot of the current data in the queue
    return []


def test_model(file_name, data_queue, WATCH_index):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("192.168.50.4", 5000))  # Replace with Raspberry Pi's IP address

    commands = [
        "M+WATCH" + str(WATCH_index),
        "D+datasets/csv/" + file_name,
    ]

    for command in commands:
        client.sendall(command.encode())
        response = client.recv(4096).decode()
        print("Server:", response)

    # send "S", when get a "S" response from the server, start the power consumption measurement
    client.sendall(b"S")
    # power_data = get_latest_data(data_queue)

    start_data = get_latest_data(data_queue)[-1]

    response = client.recv(4096).decode()

    print("Server:", response)

    # print the response until get a "F" from the server

    while "F\n" not in response:
        response = client.recv(4096).decode()
        print(f"Server:{response} end")
    # parse time:
    time_used = response.split("Time:")[-1].split("ms")[0]
    print(f"Time used: {time_used} ms")

    power_data = get_latest_data(data_queue)
    end_data = power_data[-1]

    start_stamp = start_data[0]
    end_stamp = end_data[0]
    filtered_data = [
        data for data in power_data if data[0] >= start_stamp and data[0] <= end_stamp
    ]
    energy_consumed = (filtered_data[-1][3] - filtered_data[0][3]) / 100
    print(f"Energy consumed: {energy_consumed} J")

    sleep(1)
    # send "Q" to close the connection
    client.sendall(b"Q")

    client.close()
    return energy_consumed, time_used


def main():
    # start usb_logger in another thread
    sps = 100
    max_size = sps * 100
    data_queue = deque(maxlen=max_size)
    stop = [False]
    usb_logger_thread = threading.Thread(
        target=usb_logger.power_log, args=(data_queue, stop)
    )
    usb_logger_thread.daemon = True
    usb_logger_thread.start()
    sleep(5)

    # for all file in "datasets/csv" folder
    # for all index from 0 to 38
    # then save result to a  file

    result_file_name = "pi_watch_results.csv"

    result_file = open(result_file_name, "w")
    result_file.write("file_name, index, energy_consumed(J), time_used(ms)\n")

    for index in range(39):
        # list all files in the folder
        # get the file name
        for file_name in os.listdir("datasets/csv"):
            print(f"Testing file {file_name} with index {index}")
            energy_conusmed, time_used = test_model(file_name, data_queue, index)
            result_file.write(f"{file_name}, {index}, {energy_conusmed}, {time_used}\n")

    result_file.close()
    stop[0] = True
    # close the usb_logger thread
    print("Closing usb_logger thread, join")
    usb_logger_thread.join()
    print("Closed")


if __name__ == "__main__":
    main()
