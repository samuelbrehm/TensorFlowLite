/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdio>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: minimal <tflite model>

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "minimal <tflite model>\n");
    return 1;
  }
  const char* filename = argv[1];

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Interpreter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors.
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  printf("\n\n=== Post-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // Read output buffers
  // TODO(user): Insert getting data out code.
  // Note: The buffer of the output tensor with index `i` of type T can
  // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`

  return 0;
}


// #include <iostream>
// #include <vector>
// #include "tensorflow/lite/interpreter.h"
// #include "tensorflow/lite/kernels/register.h"
// #include "tensorflow/lite/model.h"

// // Function to read image data from file (You'll need to implement this)
// // std::vector<unsigned char> ReadImageData(const std::string& filename);

// #include <vector>
// #include <string>
// #include <fstream>

// std::vector<unsigned char> ReadImageData(const std::string& filename) {
//     std::vector<unsigned char> image_data;

//     std::ifstream file(filename, std::ios::binary);
//     if (file) {
//         // Determine the size of the file
//         file.seekg(0, std::ios::end);
//         std::streamsize size = file.tellg();
//         file.seekg(0, std::ios::beg);

//         // Resize the image_data vector to hold the file's contents
//         image_data.resize(size);

//         // Read the file into the image_data vector
//         if (!file.read(reinterpret_cast<char*>(image_data.data()), size)) {
//             // Error reading file
//             image_data.clear();
//         }
//     }

//     return image_data;
// }

// int main() {
//     // Load the TensorFlow Lite model
//     const char* model_path = "/Users/samuelbrehm/Downloads/mobilenet_v1_1.0_224_quant_and_labels"; // Update with your TensorFlow Lite model path
//     std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path);
//     if (!model) {
//         std::cerr << "Error loading model: " << model_path << std::endl;
//         return 1;
//     }

//     // Create TensorFlow Lite interpreter
//     tflite::ops::builtin::BuiltinOpResolver resolver;
//     std::unique_ptr<tflite::Interpreter> interpreter;
//     tflite::InterpreterBuilder(*model, resolver)(&interpreter);
//     if (!interpreter) {
//         std::cerr << "Error creating interpreter." << std::endl;
//         return 1;
//     }

//     // Allocate tensor buffers
//     interpreter->AllocateTensors();

//     // Read image data from file (You'll need to implement this)
//     std::string image_path = "path_to_user_uploaded_image.jpg"; // Update with the user uploaded image path
//     std::vector<unsigned char> image_data = ReadImageData(image_path);

//     // Get input tensor details
//     int input_index = interpreter->inputs()[0];
//     TfLiteIntArray* input_dims = interpreter->tensor(input_index)->dims;
//     int image_height = input_dims->data[1];
//     int image_width = input_dims->data[2];
//     int image_channels = input_dims->data[3];

//     // Prepare input tensor
//     uint8_t* input_data = interpreter->typed_input_tensor<uint8_t>(0);
//     // Copy image data to input tensor buffer
//     // You may need to handle data conversion or normalization here depending on your model
//     std::memcpy(input_data, image_data.data(), image_data.size());

//     // Run inference
//     interpreter->Invoke();

//     // Process output tensors (You'll need to implement this)
//     // Output tensors contain detection scores, bounding boxes, and classes

//     return 0;
// }
