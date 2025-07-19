from tcp_server import TCPServer

"""
To run client code execute tcp_client.py file.
"""


def main():
    ''' Create a TCP Server and handle multiple clients simultaneously '''

    server = TCPServer('127.0.0.1', 8080, "menu.txt")
    server.configure_server()
    server.wait_for_client()

    server.shutdown_server()



if __name__ == '__main__':
    main()