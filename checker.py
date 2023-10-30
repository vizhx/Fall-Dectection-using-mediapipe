import socket
import cv2
import time
host = '0.0.0.0' 
port = 12345 

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.setblocking(False)
server_socket.bind((host, port))
print('connected')
while True:
    try:
        data,address = server_socket.recvfrom(1024)
        print("Received from {}: {}".format(address,data.decode('utf-8')))
    except:
        print("value not received from sensor")
    time.sleep(0.05)
    if cv2.waitKey(25) & 0xFF == ord('q'):
                break