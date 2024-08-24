# import the inference-sdk
from inference_sdk import InferenceHTTPClient

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="GLVVb7T3EYeLv8fsKt34"
)

# infer on a local image
result = CLIENT.infer("YOUR_IMAGE.jpg", model_id="test_detection-olrh0/1")
