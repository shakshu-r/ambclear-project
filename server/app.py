import threading
from inference import app, run_inference

def main():
    t = threading.Thread(target=run_inference, daemon=False)
    t.start()
    t.join()

if __name__ == "__main__":
    main()
