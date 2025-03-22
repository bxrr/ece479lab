import cv2
# import picamera
import numpy as np
from mtcnn_cv2 import MTCNN
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# import tflite_runtime.interpreter as tflite
import tensorflow as tf

# def capture_image():
#     # Instrctor note: this can be directly taken from the PiCamera documentation
#     # Create the in-memory stream
#     stream = io.BytesIO()
#     with picamera.PiCamera() as camera:
#         camera.capture(stream, format='jpeg')
        
#     # Construct a numpy array from the stream
#     data = np.frombuffer(stream.getvalue(), dtype=np.uint8)
    
#     # "Decode" the image from the array, preserving colour
#     image = cv2.imdecode(data, 1)
    
#     # OpenCV returns an array with data in BGR order. 
#     # The following code invert the order of the last dimension.
#     image = image[:, :, ::-1]
#     return image

def detect_and_crop(mtcnn, image):
    detection = mtcnn.detect_faces(image)[0]
    #TODO
    x,y,w,h = detection['box']
    box = image[x:x+w,y:y+h]
    xs = box.shape[0] 
    ys = box.shape[1]
    xp = int(xs*0.1)
    yp = int(ys*0.1)
    cropped_image = np.ones((xs + 2*xp, ys + 2*yp))
    cropped_image[xp:-xp, yp:-yp] = box
    return cropped_image

def show_bounding_box(image, bounding_box):
    x1, y1, w, h = bounding_box
    fig, ax = plt.subplots(1,1)
    ax.imshow(image)
    ax.add_patch(Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none'))
    plt.show()
    return

# preprocessing function provided to the students
def pre_process(face, required_size=(160, 160)):
    ret = cv2.resize(face, required_size)
    ret = ret.astype('float32')
    mean, std = ret.mean(), ret.std()
    ret = (ret - mean) / std
    return ret

def run_model(model, face):
# students will need to fill in the following function
    #TODO
    output_details = model.get_output_details()
    input_details = model.get_input_details()
    model.set_tensor(input_details[0]['index'], face)
    model.invoke()
    output_data = model.get_tensor(output_details[0]['index'])
    return output_data


tfl_file = "./code/resnet.tflite"
interpreter = tf.lite.Interpreter(model_path=tfl_file)
interpreter.allocate_tensors()

# process the image of the first person
input()
# 1. Read the image
mtcnn = MTCNN()
# image = capture_image()
# 2. Detect and Crop
image1 = load_image('testface1.jpg')
image1 = np.array(image1)
image2 = load_image('testface2.jpg')
image2 = np.array(image2)

cropped_image = detect_and_crop(mtcnn, image1)
# 3. Proprocess
processed_image = pre_process(cropped_image)
# 4. Run the model
print(processed_image)
data1 = run_model(interpreter, processed_image)
print(data1)

# process the image of the second person
input()
# 1. Read the image
mtcnn = MTCNN()
# image = capture_image()
# 2. Detect and Crop
cropped_image = detect_and_crop(MTCNN, image2)
# 3. Proprocess
processed_image = pre_process(cropped_image)
# 4. Run the model
data2 = run_model(interpreter, processed_image)
print(data2)

# Do the comparison of the distance