name: ambulance-env

version: "1.0"

docker:
  image: null

entrypoint:
  command: ["python", "inference.py"]
