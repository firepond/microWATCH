import socket


def main():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("192.168.50.4", 5000))  # Replace with Raspberry Pi's IP address

    commands = [
        "H",
        "C",
        "M+WATCH0",
        "D+datasets/csv/data.csv",  # Replace with the path to your dataset file
        "S",
    ]

    for command in commands:
        client.sendall(command.encode())
        response = client.recv(4096).decode()
        print("Server:", response)

    client.close()


if __name__ == "__main__":
    main()
