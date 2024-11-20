# deploy a python algorithm to the ESP32
import subprocess


def deploy(
    port,
    files,
):
    # exeucte the deploy command
    commands = [
        "ampy --port /dev/ttyUSB0 put microWatch.py",
        "ampy --port /dev/ttyUSB0 put distance.py",
    ]
    for command in commands:
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(output)


def main():
    # deploy_microWatch()
    # print current directory
    process = subprocess.Popen("pwd", stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output)


if __name__ == "__main__":
    main()
