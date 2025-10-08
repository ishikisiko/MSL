#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>

#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"

extern "C" {
  extern const unsigned char mnist_model_quantized_tflite[];
  extern const unsigned int mnist_model_quantized_tflite_len;
}

namespace {


tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input  = nullptr;
TfLiteTensor* output = nullptr;


constexpr int kTensorArenaSize = 128 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// MNIST 
constexpr int kImageRows = 28;
constexpr int kImageCols = 28;
constexpr int kImageSize = kImageRows * kImageCols;


const char* TypeName(TfLiteType t) {
  switch (t) {
    case kTfLiteFloat32: return "FLOAT32";
    case kTfLiteUInt8:   return "UINT8";
    case kTfLiteInt8:    return "INT8";
    case kTfLiteInt16:   return "INT16";
    case kTfLiteInt32:   return "INT32";
    default:             return "UNKNOWN";
  }
}


void MakeDummyDigit(uint8_t* img) {
  std::memset(img, 0, kImageSize);
  for (int r = 8; r < 20; ++r) {
    for (int c = 10; c < 18; ++c) {
      img[r * kImageCols + c] = 200; // 一个小方块
    }
  }
}

// 
void PrintTensorInfo(const TfLiteTensor* t, const char* tag) {
  if (!t) return;
  std::printf("%s:\n", tag);
  if (t->dims && t->dims->size > 0) {
    std::printf("  dims->size: %d\n", t->dims->size);
    for (int i = 0; i < t->dims->size; ++i) {
      std::printf("  dims->data[%d]: %d\n", i, t->dims->data[i]);
    }
  }
  std::printf("  type: %d (%s)\n", static_cast<int>(t->type), TypeName(t->type));

  std::printf("  quant: scale=%g, zero_point=%d\n",
              static_cast<double>(t->params.scale),
              static_cast<int>(t->params.zero_point));
}

// uint8/int8
template<typename Q>
inline Q Quantize(float real_val, float scale, int32_t zero_point);

template<>
inline uint8_t Quantize<uint8_t>(float real_val, float scale, int32_t zero_point) {
  int32_t q = static_cast<int32_t>(std::round(real_val / scale)) + zero_point;
  q = std::min(255, std::max(0, q));
  return static_cast<uint8_t>(q);
}

template<>
inline int8_t Quantize<int8_t>(float real_val, float scale, int32_t zero_point) {
  int32_t q = static_cast<int32_t>(std::round(real_val / scale)) + zero_point;
  q = std::min(127, std::max(-128, q));
  return static_cast<int8_t>(q);
}

bool SetupModel() {
  // 1)  FlatBuffer
  model = tflite::GetModel(mnist_model_quantized_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema %d not equal to supported %d.",
                           model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  //  （Conv2D, MaxPool2D, Reshape, FullyConnected）

  tflite::MicroMutableOpResolver<6> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();   
  resolver.AddMaxPool2D();
  resolver.AddReshape();
  resolver.AddFullyConnected();
  resolver.AddQuantize();          
  // resolver.AddSoftmax();


  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize /*,
      nullptr, nullptr*/);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return false;
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  std::puts("Model input:");
  PrintTensorInfo(input,  "  [input]");
  std::puts("Model output:");
  PrintTensorInfo(output, "  [output]");
  std::printf("Arena used bytes: %u\n",
              static_cast<unsigned>(interpreter->arena_used_bytes()));
  return true;
}


bool FillInput(const uint8_t* image_u8) {
  if (!input) return false;

  switch (input->type) {
    case kTfLiteFloat32: {
      float* dst = input->data.f;
      for (int i = 0; i < kImageSize; ++i) {
        dst[i] = static_cast<float>(image_u8[i]) / 255.0f;
      }
      return true;
    }
    case kTfLiteUInt8: {
      const float   scale = input->params.scale;
      const int32_t zp    = input->params.zero_point;
      uint8_t* dst = input->data.uint8;
      for (int i = 0; i < kImageSize; ++i) {
        float real = static_cast<float>(image_u8[i]) / 255.0f;
      }
      return true;
    }
    case kTfLiteInt8: {
      const float   scale = input->params.scale;
      const int32_t zp    = input->params.zero_point; 
      int8_t* dst = input->data.int8;
      for (int i = 0; i < kImageSize; ++i) {
        float real = static_cast<float>(image_u8[i]) / 255.0f;
        dst[i] = Quantize<int8_t>(real, scale, zp);
      }
      return true;
    }
    default:
      error_reporter->Report("Unsupported input tensor type: %d", input->type);
      return false;
  }
}

int ReadPrediction() {
  if (!output) return -1;

  int argmax = 0;

  switch (output->type) {
    case kTfLiteFloat32: {
      float* p = output->data.f;
      for (int i = 1; i < 10; ++i) if (p[i] > p[argmax]) argmax = i;
      return argmax;
    }
    case kTfLiteUInt8: {
      const uint8_t* p = output->data.uint8;
      for (int i = 1; i < 10; ++i) if (p[i] > p[argmax]) argmax = i;
      return argmax;
    }
    case kTfLiteInt8: {
      const int8_t* p = output->data.int8;
      for (int i = 1; i < 10; ++i) if (p[i] > p[argmax]) argmax = i;
      return argmax;
    }
    default:
      error_reporter->Report("Unsupported output tensor type: %d", output->type);
      return -1;
  }
}

} // namespace

int main() {
  std::puts("TensorFlow Lite Micro MNIST Inference\n========================================");

  if (!SetupModel()) {
    std::puts("Failed to setup model!");
    return 1;
  }

  uint8_t image[kImageSize];
  MakeDummyDigit(image);

  if (!FillInput(image)) {
    std::puts("Fill input failed.");
    return 1;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    std::puts("Invoke failed!");
    return 1;
  }

  int pred = ReadPrediction();
  if (pred < 0) {
    std::puts("Read prediction failed.");
    return 1;
  }

  std::printf("Predicted digit: %d\n", pred);
  return 0;
}
