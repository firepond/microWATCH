from collections import deque
import os
import threading
import time
import numpy as np
import serial
from time import sleep
import usb_logger


def get_latest_data(data_queue):
    if data_queue:  # Check if the queue is not empty
        return list(data_queue)  # Return a snapshot of the current data in the queue
    return []


def get_line(ser):
    # read the next non-empty line, return it, if not available return ""
    line = ""
    while line == "" and ser.in_waiting != 0:
        # read and decode from bytes to string
        line = ser.readline(ser.in_waiting or 1).decode("utf-8")
        # remove leading and trailing whitespaces
        line = line.strip()
        print("Device: " + line)
    return line


def get_lines(ser):
    lines = []
    while ser.in_waiting != 0:
        # read and decode from bytes to string
        line = ser.readline(ser.in_waiting or 1).decode("utf-8")
        # remove leading and trailing whitespaces
        line = line.strip()
        print("Device: " + line)
        lines.append(line)
    return lines


def test_model(ser, data_queue, model_name="WATCH0", data_name="apple.csv"):
    # send "A\r" unitl get a ">" response of ready to receive command, timeout =10s
    start_time = time.time()
    ser.write(b"A\r")

    while True:
        sleep(0.5)
        line = get_line(ser)
        if ">" in line:
            # device is ready
            break
        ser.write(b"A\r")
        if time.time() - start_time > 10:
            print("Error: Device not ready")
            ser.close()
            exit()

    # eat all the available data
    lines = get_lines(ser)

    # send M+ModelName
    message = "M+" + model_name + "\r"
    ser.write(bytes(message, "utf-8"))
    sleep(3)

    # read all the available data, check if the model name is set
    lines = get_lines(ser)
    model_set = False
    for line in lines:
        if "M:" in line:
            print("Model name set")
            model_set = True
            break
    if not model_set:
        print("Error: Model name not set")
        ser.close()
        exit()

    # send D+DataName\n
    message = "D+csv/" + data_name + "\r"
    ser.write(bytes(message, "utf-8"))
    time.sleep(6)
    # wait for response: D:DataName and ">" for ready to receive command
    lines = get_lines(ser)
    data_set = False
    for line in lines:
        if "D:" in line:
            print("Data set")
            data_set = True
            break
    if not data_set:
        print("Error: Data not set")
        ser.close()
        exit()

    started = False
    # send "S" to start the model
    ser.write(b"S\r")
    # constantlly check if the model is ready to start until get a "S" response, timeout = 10s

    read_start_time = time.time()
    while True:
        response = get_line(ser)
        if "S" in response:
            start_time = time.time()
            power_data = get_latest_data(data_queue)
            start_data = power_data[-1]
            break
        if time.time() - read_start_time > 10:
            print("Error: Model not ready")
            ser.close()
            exit()

    run_count = 10
    # read unitl get a "F" for model done or 'E' for error
    while True:
        response = get_line(ser)
        if "mem" in response:
            # got memory error, print the error and raise an exception
            print("Error: Memory error")
            raise Exception("Memory error")

        if "F" in response:
            end_time = time.time()
            power_data = get_latest_data(data_queue)
            end_data = power_data[-1]
            if response[1] == ":":
                run_count = int(response.split("F:")[-1])
            break
        if "E" in response:
            print("Error: Model failed")
            ser.close()
            exit()
    # read the time used from the serial
    response = get_line(ser)
    print(response)
    # get device time used
    device_time = response.split("Time:")[-1].split("ms")[0]
    # print(f"Time used: {end_time - start_time} s")
    time_in_ms = (end_time - start_time) * 1000 / run_count  # 10 times average
    start_stamp = start_data[0]
    end_stamp = end_data[0]
    filtered_data = [
        data for data in power_data if data[0] >= start_stamp and data[0] <= end_stamp
    ]
    energy_consumed = (filtered_data[-1][3] - filtered_data[0][3]) / run_count
    # print(f"Energy consumed: {energy_consumed} J")
    return energy_consumed, device_time


def main():
    # start usb_logger in another thread
    sps = 100
    max_size = sps * 1000
    data_queue = deque(maxlen=max_size)
    stop = [False]
    usb_logger_thread = threading.Thread(
        target=usb_logger.power_log, args=(data_queue, stop)
    )
    usb_logger_thread.daemon = True
    usb_logger_thread.start()

    ser = serial.Serial()
    ser.baudrate = 115200
    ser.port = "/dev/ttyUSB0"
    ser.open()
    print("Testing on " + ser.name)
    # for all files in "datasets/csv" folder and for all WATCH versons in [0-38]
    # test the model and save the result to a file
    results_file = open("esp_bocpd.csv", "w")
    # results_file = open("esp_cusum_results.csv", "w")
    results_file.write("file_name, index, energy_consumed(J), time_used(ms)\n")

    # for index in range(39):
    for file_name in os.listdir("datasets/csv"):
        # skip multivariate datasets
        data = np.loadtxt("datasets/csv/" + file_name, delimiter=",")
        if len(data.shape) > 1:
            print(f"Skipping {file_name}")
            continue
        try:
            energy, time_ms = test_model(ser, data_queue, "BOCPD", file_name)
        except:
            print(f"Error: {file_name}")
            continue
        print(f"Energy consumed: {energy} J")
        print(f"Time used: {time_ms} ms")
        results_file.write(f"{file_name}, {energy}, {time_ms}\n")
        results_file.flush()

    # uni_mode = True
    # model_name = "CUSUM"
    # for file_name in os.listdir("datasets/csv"):
    #     print(f"Testing file {file_name}")
    #     if uni_mode:
    #         # skip if  the dataset is multivariate
    #         data = np.loadtxt("datasets/csv/" + file_name, delimiter=",")
    #         if len(data.shape) > 1:
    #             print(f"Skipping {file_name}")
    #             continue

    # try:
    #     energy, time_ms = test_model(ser, data_queue, model_name, file_name)
    # except:
    #     print(f"Error: {file_name}")
    #     continue

    results_file.close()
    ser.close()

    stop[0] = True
    # close the usb_logger thread
    print("Closing usb_logger thread, join")
    usb_logger_thread.join()
    print("Closed")


if __name__ == "__main__":
    main()
