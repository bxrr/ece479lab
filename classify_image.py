# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Example using PyCoral to classify a given image using an Edge TPU.

To run this code, you must attach an Edge TPU attached to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.

Example usage:
```
bash examples/install_requirements.sh classify_image.py

python3 examples/classify_image.py \
  --model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite  \
  --labels test_data/inat_bird_labels.txt \
  --input test_data/parrot.jpg
```
"""

import argparse
import time

import numpy as np
from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter, run_inference


def main():
  image = Image.open('pants_test.jpg').convert('L').resize((28,28), Image.LANCZOS)
  image = np.resize(image, (784, 1))
  image = image.astype(np.int8)
  
  interpreter = make_interpreter('fullint.tflite')
  interpreter.allocate_tensors()

  params = common.input_details(interpreter, 'quantization_parameters')
  scale = params['scales']
  zero_point = params['zero_points']

  # Run inference
  print('----INFERENCE TIME----')
  print('Note: The first inference on Edge TPU is slow because it includes',
        'loading the model into Edge TPU memory.')
  
  start = time.perf_counter()
  run_inference(interpreter, image)
  inference_time = time.perf_counter() - start
  classes = classify.get_classes(interpreter, 1, 0)
  print('%.1fms' % (inference_time * 1000))
  
  output_details = interpreter.get_output_details()
  output_data = interpreter.get_tensor(output_details[0]['index'])
  print(output_data)

  print('-------RESULTS--------')
  for c in classes:
    print('%s: %.5f', c.score)


if __name__ == '__main__':
  main()
