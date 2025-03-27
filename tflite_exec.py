import time
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

def get_scores(interpreter):
  """Gets the output (all scores) from a classification model, dequantizing it if necessary.

  Args:
    interpreter: The ``tf.lite.Interpreter`` to query for output.

  Returns:
    The output tensor (flattened and dequantized) as :obj:`numpy.array`.
  """
  output_details = interpreter.get_output_details()[0]
  output_data = interpreter.tensor(output_details['index'])().flatten()

  if np.issubdtype(output_details['dtype'], np.integer):
    scale, zero_point = output_details['quantization']
    # Always convert to np.int64 to avoid overflow on subtraction.
    return scale * (output_data.astype(np.int64) - zero_point)

  return output_data.copy()

# load the TFLite model and allocate tensors
interpreter = tflite.Interpreter('L2P2model_fq.tflite')
interpreter.allocate_tensors()

# get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# test the model on random input data
input_shape = input_details[0]['shape']
# input_data = np.array(np.random.random_sample(input_shape) * 255, dtype=np.int8)
image = Image.open('MNIST2.png').convert('L').resize((28,28), Image.LANCZOS)
image = np.resize(image, (input_shape))
image = image.astype(np.uint8)
#print(image)
test_images = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')

#interpreter.set_tensor(input_details[0]['index'], image)
correct_inf = 0

# run the model
start = time.perf_counter()
for i in range(test_images.shape[0]):
    currimg = np.resize(test_images[i], (input_shape))
    currimg = currimg.astype(np.uint8)
    interpreter.set_tensor(input_details[0]['index'], currimg)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    ind = np.argmax(output_data)
    if(ind == test_labels[i]):
        correct_inf += 1
inference_time = time.perf_counter() - start
print('Full-integer quantization with no USB Accelerator')
print('Run Time: %.1fms' % (inference_time * 1000))
print("Accuracy: ", correct_inf/(1.0*test_images.shape[0]))

# get_tensor() returns copy of the tensor data
output_data = interpreter.get_tensor(output_details[0]['index'])

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'AnkleBoot']

#ind = np.argmax(output_data)
#scores = get_scores(interpreter)
#print(scores)
#print(labels[ind], ":", scores[ind])