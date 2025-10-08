// mnist_inference_diag.cc
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cinttypes>

#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"

// ===== 模型数组（由 xxd -i 生成并编译进项目） =====
extern "C" {
  extern const unsigned char mnist_model_quantized_tflite[];
  extern const unsigned int  mnist_model_quantized_tflite_len;
}

// ===== 真实 MNIST 5 样例头文件（脚本生成） =====
#include "mnist_5_samples.h"  // 提供 sample_0..4、kSamples、kSampleLabels、kNumSamples

namespace {

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input  = nullptr;
TfLiteTensor* output = nullptr;

// ---------- arena 与 guard ----------
constexpr size_t kTensorArenaSize = 1024 * 1024; // 1 MB
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// 哨兵（紧跟 arena，检查是否被溢出写坏）
static uint8_t tensor_arena_guard[64] = { 0x55 };

// MNIST 输入尺寸
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

void PrintBytes(const void* p, size_t n) {
  const unsigned char* b = reinterpret_cast<const unsigned char*>(p);
  for (size_t i = 0; i < n; ++i) {
    std::printf("%02X ", b[i]);
    if ((i+1) % 16 == 0) std::printf("\n");
  }
  if (n % 16) std::printf("\n");
}

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
  std::printf("  bytes: %zu\n", t->bytes);
  // data pointer (raw) is useful
  std::printf("  data.raw ptr: %p\n", t->data.raw);
}

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

// Print some useful addresses and guard bytes
void DumpArenaAndGuardState(const char* when) {
  std::printf("DBG: [%s] tensor_arena start=%p end=%p size=%zu\n",
              when, (void*)tensor_arena, (void*)(tensor_arena + kTensorArenaSize), kTensorArenaSize);
  std::printf("DBG: [%s] tensor_arena_guard at %p (64 bytes):\n", when, (void*)tensor_arena_guard);
  PrintBytes(tensor_arena_guard, sizeof(tensor_arena_guard));
}

// SetupModel: 打印 operator_codes，并注册常见算子
bool SetupModel() {
  model = tflite::GetModel(mnist_model_quantized_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema %d not equal to supported %d.",
                           model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  std::puts("DBG: Model operator_codes:");
  auto opcodes = model->operator_codes();
  if (opcodes) {
    for (size_t i = 0; i < opcodes->Length(); ++i) {
      auto oc = opcodes->Get(i);
      const char* custom = "<null>";
      if (oc->custom_code()) custom = oc->custom_code()->c_str();
      std::printf("  opcode[%zu]: builtin_code=%d custom=%s\n",
                  i, static_cast<int>(oc->builtin_code()), custom);
    }
  } else {
    std::puts("  (no operator_codes)");
  }

  auto subgraphs = model->subgraphs();
  if (subgraphs && subgraphs->Length() > 0) {
    auto sg = subgraphs->Get(0);
    auto ops = sg->operators();
    std::puts("DBG: Subgraph[0] operators (opcode_index per op):");
    if (ops) {
      for (size_t i = 0; i < ops->Length(); ++i) {
        auto op = ops->Get(i);
        std::printf("  op[%zu]: opcode_index=%d\n", i, op->opcode_index());
      }
    }
  }

  // 在调用 AllocateTensors() 前，用可识别模式填充 arena（有助于排查）
  std::memset(tensor_arena, 0xCD, kTensorArenaSize);
  std::puts("DBG: tensor_arena filled with 0xCD (pattern)");
  DumpArenaAndGuardState("after memset");

  // 注册常见算子
  tflite::MicroMutableOpResolver<16> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddAveragePool2D();
  resolver.AddMaxPool2D();
  resolver.AddReshape();
  resolver.AddFullyConnected();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddSoftmax();
  resolver.AddRelu();
  resolver.AddRelu6();
  resolver.AddAdd();
  resolver.AddMul();
  resolver.AddPad();
  resolver.AddConcatenation();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, static_cast<int>(kTensorArenaSize));
  interpreter = &static_interpreter;

  std::printf("DBG: calling AllocateTensors() with arena %zu bytes...\n", kTensorArenaSize);
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
  std::printf("Arena used bytes: %u\n", static_cast<unsigned>(interpreter->arena_used_bytes()));

  DumpArenaAndGuardState("after AllocateTensors");
  // Also dump a small range at the end of the arena (to see if allocator scribbled near end)
  const size_t tail_dump = 64;
  uint8_t* arena_end_ptr = tensor_arena + kTensorArenaSize;
  std::printf("DBG: tail %zu bytes before arena end (addresses %p..%p):\n",
              tail_dump, (void*)(arena_end_ptr - tail_dump), (void*)(arena_end_ptr - 1));
  PrintBytes(arena_end_ptr - tail_dump, tail_dump);

  return true;
}

// 填充输入（输入数据为 0..255 uint8_t）
bool FillInputFromImageU8(const uint8_t* image_u8) {
  if (!input) {
    std::fprintf(stderr, "DBG: FillInput failed: input == nullptr\n");
    return false;
  }

  if (input->bytes < static_cast<size_t>(kImageSize)) {
    std::fprintf(stderr, "DBG: FillInput: input->bytes=%zu < expected %d\n", input->bytes, kImageSize);
    return false;
  }

  std::printf("DBG: FillInput: input->type=%d (%s), bytes=%zu\n",
              static_cast<int>(input->type), TypeName(input->type), input->bytes);
  std::printf("DBG: input->data.raw = %p\n", input->data.raw);

  // Check whether input->data.raw lies within tensor_arena
  uintptr_t arena_start = reinterpret_cast<uintptr_t>(tensor_arena);
  uintptr_t arena_end   = reinterpret_cast<uintptr_t>(tensor_arena + kTensorArenaSize);
  uintptr_t input_ptr   = reinterpret_cast<uintptr_t>(input->data.raw);
  std::printf("DBG: arena range: %p .. %p\n", (void*)arena_start, (void*)arena_end);
  std::printf("DBG: input data ptr: %p (as uintptr 0x%" PRIxPTR ")\n", (void*)input_ptr, input_ptr);

  bool in_arena = (input_ptr >= arena_start) && (input_ptr < arena_end);
  std::printf("DBG: input->data.raw inside arena? %d\n", in_arena ? 1 : 0);

  // Check whether writing kImageSize bytes will overflow arena
  uintptr_t input_end = input_ptr + static_cast<uintptr_t>(kImageSize);
  std::printf("DBG: input_end ptr would be %p\n", (void*)input_end);
  bool would_overflow = (input_end > arena_end);
  std::printf("DBG: writing %d bytes will overflow arena? %d\n", kImageSize, would_overflow ? 1 : 0);

  switch (input->type) {
    case kTfLiteFloat32: {
      float* dst = input->data.f;
      for (int i = 0; i < kImageSize; ++i) dst[i] = image_u8[i] / 255.0f;
      std::printf("DBG: FillInput float32 first8: ");
      for (int i = 0; i < 8; ++i) std::printf("%g ", dst[i]);
      std::printf("\n");
      return true;
    }
    case kTfLiteUInt8: {
      const float scale = input->params.scale;
      const int32_t zp = input->params.zero_point;
      uint8_t* dst = input->data.uint8;
      for (int i = 0; i < kImageSize; ++i) {
        float real = image_u8[i] / 255.0f;
        dst[i] = Quantize<uint8_t>(real, scale, zp);
      }
      std::printf("DBG: FillInput uint8 first8: ");
      for (int i = 0; i < 8; ++i) std::printf("%u ", static_cast<unsigned>(dst[i]));
      std::printf("\n");
      return true;
    }
    case kTfLiteInt8: {
      const float scale = input->params.scale;
      const int32_t zp = input->params.zero_point;
      int8_t* dst = input->data.int8;
      for (int i = 0; i < kImageSize; ++i) {
        float real = image_u8[i] / 255.0f;
        dst[i] = Quantize<int8_t>(real, scale, zp);
      }
      std::printf("DBG: FillInput int8 first8: ");
      for (int i = 0; i < 8; ++i) std::printf("%d ", static_cast<int>(dst[i]));
      std::printf("\n");
      return true;
    }
    default:
      error_reporter->Report("Unsupported input type: %d", input->type);
      return false;
  }
}

int Argmax10() {
  if (!output) return -1;

  std::printf("DBG: Argmax10: out.type=%d (%s), out.bytes=%zu\n",
              static_cast<int>(output->type), TypeName(output->type), output->bytes);

  if (output->bytes < 10u) {
    std::fprintf(stderr, "DBG: Argmax10: output->bytes=%zu < 10\n", output->bytes);
    return -1;
  }

  int argmax = 0;
  switch (output->type) {
    case kTfLiteFloat32: {
      float* p = output->data.f;
      std::printf("DBG: output float first10: ");
      for (int i = 0; i < 10; ++i) std::printf("%g ", p[i]);
      std::printf("\n");
      for (int i = 1; i < 10; ++i) if (p[i] > p[argmax]) argmax = i;
      return argmax;
    }
    case kTfLiteUInt8: {
      const uint8_t* p = output->data.uint8;
      std::printf("DBG: output uint8 first10: ");
      for (int i = 0; i < 10; ++i) std::printf("%u ", static_cast<unsigned>(p[i]));
      std::printf("\n");
      for (int i = 1; i < 10; ++i) if (p[i] > p[argmax]) argmax = i;
      return argmax;
    }
    case kTfLiteInt8: {
      const int8_t* p = output->data.int8;
      std::printf("DBG: output int8 first10: ");
      for (int i = 0; i < 10; ++i) std::printf("%d ", static_cast<int>(p[i]));
      std::printf("\n");
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
  std::puts("TFLM MNIST Inference (5 real samples)");
  std::puts("=====================================");

  if (!SetupModel()) {
    std::puts("Failed to setup model!");
    return 1;
  }

  if (!input) {
    std::fprintf(stderr, "ERROR: input tensor is null\n");
    return 1;
  }
  if (!output) {
    std::fprintf(stderr, "ERROR: output tensor is null\n");
    return 1;
  }
  std::printf("DBG: main: input->bytes=%zu output->bytes=%zu\n", input->bytes, output->bytes);

  // 逐个跑 5 张真实样例（由 mnist_5_samples.h 提供）
  for (int idx = 0; idx < kNumSamples; ++idx) {
    std::printf("\n=== DBG: start sample %d ===\n", idx);
    const uint8_t* img = kSamples[idx];
    int gt = kSampleLabels[idx];

    std::printf("DBG: before FillInput idx=%d\n", idx);
    // Dump guard before FillInput too
    DumpArenaAndGuardState("before FillInput");
    bool ok = FillInputFromImageU8(img);
    std::printf("DBG: after  FillInput idx=%d ok=%d\n", idx, ok ? 1 : 0);
    DumpArenaAndGuardState("after FillInput");

    if (!ok) {
      std::printf("DBG: skipping sample %d due to FillInput failure\n", idx);
      continue;
    }

    // 在 Invoke 前检查 guard
    bool guard_ok_before = true;
    for (size_t i = 0; i < sizeof(tensor_arena_guard); ++i) {
      if (tensor_arena_guard[i] != 0x55) { guard_ok_before = false; break; }
    }
    std::printf("DBG: guard before Invoke ok=%d\n", guard_ok_before ? 1 : 0);

    std::printf("DBG: before Invoke idx=%d\n", idx);
    TfLiteStatus st = interpreter->Invoke();
    std::printf("DBG: after  Invoke idx=%d status=%d\n", idx, (int)st);

    // 在 Invoke 后检查 guard
    bool guard_ok_after = true;
    for (size_t i = 0; i < sizeof(tensor_arena_guard); ++i) {
      if (tensor_arena_guard[i] != 0x55) { guard_ok_after = false; break; }
    }
    std::printf("DBG: guard after Invoke ok=%d\n", guard_ok_after ? 1 : 0);
    if (!guard_ok_after) {
      std::fprintf(stderr, "ERROR: tensor_arena overflow detected! guard corrupted.\n");
      DumpArenaAndGuardState("on guard corruption");
      return 1;
    }

    if (st != kTfLiteOk) {
      std::printf("DBG: Invoke returned error for sample %d\n", idx);
      continue;
    }

    std::printf("DBG: before Argmax idx=%d\n", idx);
    int pred = Argmax10();
    std::printf("DBG: after  Argmax idx=%d pred=%d\n", idx, pred);

    std::printf("[Sample %d] GT=%d  Pred=%d\n", idx, gt, pred);
  }

  std::puts("Done.");
  return 0;
}
