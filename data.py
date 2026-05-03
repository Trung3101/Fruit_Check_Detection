from roboflow import Roboflow
import os

rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY", "your_api_key_here"))
project = rf.workspace("immortal-tower").project("fruit-dataset-ctdky")
version = project.version(12)
dataset = version.download("yolo26")
                