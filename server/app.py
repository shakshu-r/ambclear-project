import threading
from inference import app, run_inference

def main():
    t = threading.Thread(target=run_inference, daemon=True)
    t.start()

if __name__ == "__main__":
    main()
