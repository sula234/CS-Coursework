import socket
import struct, random, json
from datetime import datetime



REQUEST_TYPE = {
    "MENU": 10, 
    "ORDR": 13,
    "PAYM": 17
}


class TCPClient():

    def __init__(self, IP: str, port: int) -> None:
        self.id = random.randint(0, 10000)
        self.IP = IP
        self.port = port
        self.client = None
        self.last_order_request = None
        self.last_total_price = None

    def printwt(self, msg):
        ''' Print message with current date and time '''

        current_date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'[{current_date_time}][CLIENT] {msg}')



    def create_order(self, body):
        
        while True:
            dish = input("Hello! Write what will you order (to submit order type 'q' ): ")
            dish = dish.replace(" ", "")
            if dish.lower() == 'q':
                break

            qty = input("Quantity: ")
            qty = qty.replace(" ", "")
            
            body[dish] = qty


    def create_payment(self, body):
        
        self.printwt(f"Starting payment creation for ORDER Request: {self.last_order_request}")
        name = input("Please enter your name: ")
        adress = input("Please enter your adress: ")
        card_num = input("Please enter your card number: ")
        total_price = input("Please enter your total price: ")

        body["name"] = name
        body["adress"] = adress
        body["card_num"] = card_num
        body["total_price"] = total_price
            


    def connect_to_server(self):
        ''' Connect to existing server '''

        #
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((self.IP, self.port))
        self.printwt("Connection!")

        # Receive data from the server
        received_data = self.client.recv(1024).decode('utf-8')

        # Print the received message
        self.printwt(f'Intitial received message from server: {received_data}')    

    def close_connection(self) -> None:
        self.client.close()

    
    def send_request(self, req_type):

        body={}
        # generate random request id 
        if req_type == "PAYM":
            request_id = self.last_order_request
        else:
            request_id = random.randint(0, 4294967295)
        
        # specify logic for req type
        if req_type == "MENU":
            self.printwt("Send request to get MENU")

        elif req_type == "ORDR":
            self.create_order(body)

            if len(body) != 0:
                self.printwt("Send request to create ORDER")
            else:
                self.printwt("To make a payment you need to create non empty order.")
                self.printwt("Request rejected at client level")
                return -1

        elif req_type == "PAYM":
            if self.last_order_request:
                self.create_payment(body)
                self.printwt("Send request to create PAYMENT")
            else:
                self.printwt("To make a payment you need to create order.")
                self.printwt("Request rejected at client level")
                return -1
        
        # pack info data
        package = struct.pack("I B", request_id, REQUEST_TYPE[req_type])

        # dump body data 
        body = json.dumps(body).encode("utf-8")
        

        # merge body and info data
        package += body
        
        # Send the packed data to the server
        self.client.sendall(package)

        # Receive response from the server
        response_package = self.client.recv(1024)
        self.handle_response(response_package)

    
    def handle_response(self, response_package):

        # get information part from package
        info_part = response_package[0:9]

        response_id, request_id, request_type = struct.unpack('I I B', info_part)

        self.printwt(f"Got Response with ID: {response_id}")
        if request_type == REQUEST_TYPE["MENU"]:
            dumped_menu = response_package[9:]

            # Decode and parse the received menu
            menu = json.loads(dumped_menu.decode('utf-8'))

            print("\n************************************************")

            print("MENU:\n************************************************")

            for dish in menu.keys():
                print(f"{dish}: {menu[dish]}")

            print("************************************************\n")

        elif request_type == REQUEST_TYPE["ORDR"]:

            # get ORDR status from value 
            ORDR_status = struct.unpack('B', response_package[9:10])[0]

            if ORDR_status == 1:

                total_price = struct.unpack('I', response_package[10:14])[0]
                self.printwt(f"ORDER was successful! Total price: {total_price}")
                self.last_order_request = request_id
            else:
                wrong_item = response_package[10:].decode()
                self.printwt(f"ORDER was not successful. Wrong item: {wrong_item}")
        
        elif request_type == REQUEST_TYPE["PAYM"]:

            # get ORDR status from value 
            PAYM_status = struct.unpack('B', response_package[9:10])[0]

            if PAYM_status == 1:

                #thanks_message = struct.unpack('I', response_package[10:14])[0]
                self.printwt(f"PAYMENT was successful! order request id: {request_id}")
                self.last_order_request = None
            else:
                
                self.printwt(f"PAYMENT was not successful! order request id: {request_id}")




client = TCPClient('127.0.0.1', 8080)
client.connect_to_server()
while True:        
    req_type = input("Please enter, request type (to exit enter 'q' )\nRequest types:\n1)'MENU': get menu\n2)'ORDR': create and submit order\n3)'PAYM': make a payment\n: ")
    if req_type.lower() == 'q':
        break
    
    try:
        client.send_request(req_type.upper())

    except KeyError:
        print("\n************************************************")
        print("Please enter valid request type! (PAYM, MENU or ORDR)")
        print("************************************************\n")

client.close_connection()