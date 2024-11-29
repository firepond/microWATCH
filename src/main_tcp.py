import socket
import threading
import time
import numpy as np
from algorithms import micro_watch as microwatch

model_name = ""
model_set = False

dataset_path = ""
data = []
params = []
model_set = False
data_set = False

start = False

def get_data(file_path):
    data = np.loadtxt(file_path, delimiter=",")
    return data




def handle_client(client_socket):
    model_name = ""
    model_set = False
    dataset_path = ""
    data = []
    data_set = False

    def get_data(file_path):
        return np.loadtxt(file_path, delimiter=",")

    while True:
        try:
            command = client_socket.recv(1024).decode().strip()
            if not command:
                continue
            
            print(command)
            # if "Q", close the connection and reset to prepare for new task
            if command =="Q":
                break
            
            if command == "H":
                response = "hello\nboard: Raspberry Pi\nlanguage: Python\n"
            elif command == "C":
                response = "H\nA\nM\nD\nS\n"
            elif command.startswith("M+"):
                model_name = command[2:]
                model_set = True
                if "WATCH" in model_name:
                    # parse the int from the model name
                    try:
                        distance_index = int(model_name.split("WATCH")[-1])
                    except:
                        print("E")
                        continue
                    watch = microwatch.microWATCH()
                    watch.metric = microwatch.distance_measures[distance_index]
                    model_set = True
                    print("M:" + model_name)
                    model_name = "WATCH"
                    response = f"M:{model_name}\n"

            elif command.startswith("D+"):
                if not model_set:
                    response = "E\n"
                else:
                    dataset_path = command[2:]
                    print(dataset_path)
                    try:
                        data = get_data(dataset_path)
                        print(data.shape)
                    except:
                        response = f"E\n{str(e)}\n"
                        continue

                    # load params
                    try:
                    # params = get_params(dataset_path, distance_index)
                        print("reading params")
                        watch.set_params(distance_index, dataset_path)
                    except:
                        print("load params error")
                        print("E")
                        continue
                    
                    data_set = True
                    print("D:" + dataset_path)
                        
                    response = f"D:{dataset_path}\n"
                    
            elif command == "S":
                if not model_set or not data_set:
                    response = "E\n"
                else:
                    client_socket.sendall("S".encode())
                    start_time = time.time()
                    # Simulate model processing (replace with real logic)
                    for i in range(100):
                        location  = watch.detect(data)
                        watch.reinit()
                    end_time = time.time()
                    response = f"F\nTime: {(end_time - start_time) * 10:.2f} ms\n"
            else:
                response = "E\n"
            
            client_socket.sendall(response.encode())
        except Exception as e:
            client_socket.sendall(b"E\n")
            print(f"Error: {str(e)}")
            break
    client_socket.close()

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("0.0.0.0", 5000))
    server.listen(5)
    print("Server listening on port 5000...")
    
    while True:
        client_socket, addr = server.accept()
        print(f"Accepted connection from {addr}")
        client_handler = threading.Thread(target=handle_client, args=(client_socket,))
        client_handler.start()

if __name__ == "__main__":
    main()
