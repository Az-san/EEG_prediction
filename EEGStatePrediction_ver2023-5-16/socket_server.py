import socket

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
HOST = "172.17.6.135"
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

# server
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:  # main socket loop
        print(f"Connected by {addr}")
        while True:
            data = conn.recv(1024)
            
            print(data)
            
print('hellow')

