#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/tools/gen_op_registration.h>
#include "/opt/homebrew/Cellar/opencv/4.9.0_2/include/opencv4/opencv2/opencv.hpp"
#include <iostream>
#include <memory>

int main(int argc, char* argv[]) {
//     if (argc != 2) {
//         std::cerr << "Usage: object_detection_tflite <image_file>" << std::endl;
//         return 1;
//     }

//     const char* image_path = argv[1];
//     cv::Mat original_image = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::Mat original_image = cv::imread("images/pexels-pixabay-46798.jpg", cv::IMREAD_COLOR);
    if (original_image.empty()) {
        std::cerr << "Cannot read input image" << std::endl;
        return 1;
    }

    cv::Mat input_image;
    cv::resize(original_image, input_image, cv::Size(300, 300));
    cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);

    input_image.convertTo(input_image, CV_32F, 1.0 / 255);

    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("models/mobilenet_v1_1.0_224_quant_and_labels/mobilenet_v1_1.0_224_quant.tflite");
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors!" << std::endl;
        return 1;
    }

    float* input_tensor = interpreter->typed_input_tensor<float>(0);
    // Copy image data to input tensor
    memcpy(input_tensor, input_image.data, 300 * 300 * 3 * sizeof(float));

    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke interpreter!" << std::endl;
        return 1;
    }

    // Output tensor parsing and drawing results (to be implemented based on model output format)
    // Example: Extract detection boxes and labels, then draw them on the original_image

    // Show the result
    cv::imshow("Object Detection", original_image);
    cv::waitKey(0);

    return 0;
}
