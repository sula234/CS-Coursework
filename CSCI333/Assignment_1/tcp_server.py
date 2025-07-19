import socket
import threading, random
import struct, json

from datetime import datetime


ORDR_LIST = {}


class TCPServer():
    ''' 
    A  TCP Server for handling multiple clients 
    '''

    def __init__(self, 
                 host, 
                 port, 
                 menu_path
                 ):
        
        """
        Initialize a TCPServer instance.

        Attributes:
            host (str): IP adress 
            port (int): integer number for port 
            menu_path (str): path to .txt file containing menu 

        """
        self.host = host            # Host address
        self.port = port            # Host port
        self.sock = None            # Connection socket
        self.menu_path = menu_path  # relative path to menu.txt file 


    def printwt(self, msg):
        ''' 
        Print message with current date and time 
        '''

        current_date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'[{current_date_time}][SERVER] {msg}')


    def configure_server(self):
        ''' 
        Set up the server 
        '''

        # create TCP socket with IPv4 addressing
        self.printwt('Creating socket...')
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.printwt('Socket created')

        # bind server to the address
        self.printwt(f'Binding server to {self.host}:{self.port}...')
        self.sock.bind((self.host, self.port))
        self.printwt(f'Server binded to {self.host}:{self.port}')


    def wait_for_client(self):
        ''' 
        Wait for clients to connect 
        '''

        try:
            self.printwt('Listening for incoming connection')
            self.sock.listen(20) # 10 clients before server refuses connections

            while True:

                client_sock, client_address = self.sock.accept()
                self.printwt(f'Accepted connection from {client_address}')

                # Send a message to the client
                client_sock.sendall("At your service".encode('utf-8'))
                c_thread = threading.Thread(target = self.handle_client,
                                        args = (client_sock, client_address))
                c_thread.start()

        except KeyboardInterrupt:
            self.shutdown_server()


    def handle_client(self, client_sock, client_address):
        '''
        Handle the accepted client's requests 

        Attributes: 
        '''

        try:
            # client's request
            request_package = client_sock.recv(1024)
            while request_package:
                
                # get body part from package
                dumped_body = request_package[5:]

                # get information part from package
                info_part = request_package[0:5]

                # unpack header information
                request_id, request_type = struct.unpack('I B', info_part)
                self.printwt(f'[ REQUEST from {client_address} ]')

                # generate response id 
                response_id = random.randint(0, 4294967295)
                # create initial response package
                response_package = struct.pack('I', response_id) + info_part

                # parse menu from .txt file 
                menu = self.get_menu()

                # request type == MENU
                if request_type == 10:

                    # merge response with menu
                    response_package += json.dumps(menu).encode("utf-8")

                # request type == ORDR
                elif request_type == 13:
                    body = json.loads(dumped_body.decode('utf-8'))
                    is_valid, wrong_item, total_price = self.is_order_valid(body, menu)

                    if is_valid:
                        self.printwt(f"Order is valid. Client request id: {request_id}")

                        # save user request id with corresponding total price
                        ORDR_LIST[request_id] = total_price

                        
                        response_package += struct.pack('B', 1)
                        response_package += struct.pack('I', total_price)
                        

                    else:
                        self.printwt(f"Order is not valid. Item: {wrong_item}")

                        # add error code 
                        response_package += struct.pack('B', 3)

                        # encode wrong item
                        wrong_item = bytes(wrong_item, 'utf-8')

                        # add error item at the end
                        response_package += wrong_item


                # request type == PAYM
                elif request_type == 17:
                    if request_id in ORDR_LIST.keys():
                        body = json.loads(dumped_body.decode('utf-8'))
                        payment_valid = self.is_payment_valid(body, request_id)


                        if payment_valid:
                            self.printwt(f"Payment is valid for user with ORDR request id: {request_id}")
                            response_package += struct.pack('B', 1)
                            # response_package += struct.pack('', total_price)
                        else:
                            self.printwt(f"Payment is NOT valid for user with ORDR request id: {request_id}")
                            response_package += struct.pack('B', 3)
                    
                    else:
                        raise OSError



                # send response
                self.printwt(f'[ RESPONSE to {client_address} with  resposne ID:{response_id}]')
                client_sock.sendall(response_package)


                # get more data and check if client closed the connection
                request_package = client_sock.recv(1024)
            self.printwt(f'Connection closed by {client_address}')

        except OSError as err:
            self.printwt(err)

        finally:
            self.printwt(f'Closing client socket for {client_address}...')
            client_sock.close()
            self.printwt(f'Client socket closed for {client_address}')


    def shutdown_server(self):
        ''' 
        Shutdown the server 
        '''

        self.printwt('Shutting down server...')
        self.sock.close()


    def get_menu(self):
        '''
        Helper Function to get menu in dict format from .txt file
        '''
        f = open(self.menu_path, 'r')
        menu = {}

        for line in f:
            dish, price = line.strip().split(': ')
            menu[dish] = price
        
        return menu
    

    def is_order_valid(self, order: dict, menu):
        '''
        Check if ordrer is valid and calculate total prices

        Attributes:
            order (Dict[str, str]): ordered products with an name as a key and price as a value
            menu (Dict[str, str]): available menu to compare with order and find wrong item

        '''
        
        total_price = 0
        for item in order:
            if item.lower() not in menu:
                return False, item, total_price
            total_price += int(menu[item]) * int(order[item])
        return True, None, total_price


    def is_payment_valid(self, body, request_id):
        '''
        Check if ordrer is valid and calculate total prices

        Attributes:
            body (Dict[str, str]): decoded body part of the package from "PAYM" request
            request_id (int):  request id attached to the ORDR
        '''
        
        if int(body["total_price"]) == ORDR_LIST[request_id]:
            return True
        else: 
            return False




