import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np

# Load model
model = YOLO("person_vehicle_model.pt")

def detect(image):
    results = model(image)
    annotated = results[0].plot()
    return annotated

interface = gr.Interface(
    fn=detect,
    inputs=gr.Image(source="webcam", streaming=True),
    outputs="image",
    title="🚗 Person & Vehicle Detection System",
    description="""
Live detection using custom YOLO model.

⚡ For better results, test the model while traveling on the road
or in an outdoor environment where vehicles and pedestrians are visible.
""",
)

interface.launch()
