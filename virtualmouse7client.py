import socket

def main():
    HOST = input("Enter server IP address: ")
    PORT = 6000

    # Create TCP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((HOST, PORT))
            print(f"Connected to {HOST}:{PORT}. Type messages and press Enter to send. Ctrl+C to quit.")
        except Exception as e:
            print(f"Connection error: {e}")
            return

        while True:
            try:
                msg = input('> ')
                if not msg:
                    continue
                s.sendall(msg.encode())
            except KeyboardInterrupt:
                print("\nExiting client.")
                break
            except Exception as e:
                print(f"Send error: {e}")
                break

if __name__ == '__main__':
    main()
