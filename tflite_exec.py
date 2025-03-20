import time
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

# load the TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path='fullint.tflite')
interpreter.allocate_tensors()

# get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# test the model on random input data
input_shape = input_details[0]['shape']
# input_data = np.array(np.random.random_sample(input_shape) * 255, dtype=np.int8)
image = Image.open('pants_test.jpg').convert('L').resize((28,28), Image.LANCZOS)
image = np.resize(image, (input_shape))
image = image.astype(np.int8)
#print(image)
interpreter.set_tensor(input_details[0]['index'], image)

# run the model
for _ in range(5):
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    print('%.1fms' % (inference_time * 1000))


# get_tensor() returns copy of the tensor data
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'AnkleBoot']

ind = np.argmax(output_data)
print(labels[ind])
