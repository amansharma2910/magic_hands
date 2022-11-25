import os
import PIL
import cv2
import numpy as np
import tensorflow as tf


# load the model
def load_model():
    model_path = os.path.join(os.getcwd(), "model")
    model = tf.keras.models.load_model(model_path)
    return model


# function to crop the image and resize it to 256 x 256 pixels
def crop_and_resize_image(frame):
    img = PIL.Image.fromarray(frame)

    width, height = img.size
    size = min(width, height)
    
    left = (width - size)/2
    top = (height - size)/2
    right = (width + size)/2
    bottom = (height + size)/2

    # square crop the center of the image
    img = img.crop((left, top, right, bottom)).resize((256,256))
    
    img = np.array(img)
    return img


# function to generate prediction on one frame
def generate_inference_on_frame(frame, model):
    h, w, c = frame.shape
    frame = frame.reshape((1, h, w, c))
    probabilities = tf.nn.softmax(model(frame)[0]).numpy()
    if np.max(probabilities) < 0.9:
        return
    result = np.argmax(probabilities)
    return result


# driver function
def start_magic_hands(model):
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("Magic Hands")

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            # cv2.imshow("Magic Hands", frame)
            frame = crop_and_resize_image(frame)
            result = generate_inference_on_frame(frame, model)
            if result is not None:
                # instead of printing result, you can run a script based on the result
                print("Result is:", result)
                break    

            # pressing escape key will terminate the camera operation
            k = cv2.waitKey(1)
            if k%256 == 27:
                print("Esc key press detected. Closing...")
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    model = load_model()
    start_magic_hands(model)