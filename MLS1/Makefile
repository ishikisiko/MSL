TARGET = mnist_inference
CC = g++
CFLAGS = -std=c++11 -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON \
         -O3 -fno-exceptions -fno-rtti -fno-threadsafe-statics \
         -ffunction-sections -fdata-sections -fmessage-length=0

LDFLAGS = -Wl,--gc-sections -Wl,--no-export-dynamic \
          -Wl,--exclude-libs,ALL -Wl,--no-undefined

TFLM_ROOT := ./tflite-micro

INCLUDES = -I./tflite-micro/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
           -I$(TFLM_ROOT) \
           -I$(TFLM_ROOT)/third_party/gemmlowp \
           -I$(TFLM_ROOT)/third_party/ruy

SOURCES = part3_micro_deployment.cc model_data.cc
TFLM_LIB = libtensorflow-microlite.a

.PHONY: all debug asan run-gdb clean clean-all

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) $(SOURCES) $(TFLM_LIB) $(LDFLAGS) -o $(TARGET)

# ---------------- debug build (no opt, with symbols) ----------------
CFLAGS_DEBUG = -std=c++11 -g -O0 -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON \
         -fno-exceptions -fno-rtti -fno-threadsafe-statics \
         -ffunction-sections -fdata-sections -fmessage-length=0

debug: mnist_inference_dbg

mnist_inference_dbg: $(SOURCES)
	$(CC) $(CFLAGS_DEBUG) $(INCLUDES) $(SOURCES) $(TFLM_LIB) $(LDFLAGS) -o $@

# ---------------- asan build (if you want ASan diagnostics) ----------
# Note: ASan may fail if static libs were built without compatible flags.
CFLAGS_ASAN = -std=c++11 -g -O1 -fsanitize=address -fno-omit-frame-pointer \
              -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON \
              -fno-exceptions -fno-rtti -fno-threadsafe-statics \
              -ffunction-sections -fdata-sections -fmessage-length=0

LDFLAGS_ASAN = -fsanitize=address

asan: mnist_inference_asan

mnist_inference_asan: $(SOURCES)
	$(CC) $(CFLAGS_ASAN) $(INCLUDES) $(SOURCES) $(TFLM_LIB) $(LDFLAGS) $(LDFLAGS_ASAN) -o $@

# ---------------- helper: run gdb on debug binary -------------------
run-gdb: mnist_inference_dbg
	gdb ./mnist_inference_dbg

# ---------------- clean targets ------------------------------------
clean:
	rm -f $(TARGET)

clean-all: clean
	rm -f mnist_inference_dbg mnist_inference_asan
