#!/usr/bin/env python3
import csv, urllib.request, io, sys

# 你也可以改成全量集：dbdmg 的教学数据 mnist_test.csv（行很多，脚本逻辑一样）
RAW_URL = "https://raw.githubusercontent.com/makeyourownneuralnetwork/makeyourownneuralnetwork/master/mnist_dataset/mnist_test_10.csv"
OUT_HDR = "mnist_5_samples.h"
N = 5  # 取前 5 条

def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url) as r:
        return r.read().decode("utf-8")

def main():
    print(f"Downloading: {RAW_URL}")
    text = fetch_text(RAW_URL)
    rows = list(csv.reader(io.StringIO(text)))
    assert len(rows) >= N, "数据行不足"

    # 写 C 头文件
    with open(OUT_HDR, "w", encoding="utf-8") as f:
        f.write("// Auto-generated from MNIST CSV (first {} rows)\n".format(N))
        f.write("// Source: {}\n\n".format(RAW_URL))
        f.write("#pragma once\n#include <cstdint>\n\n")
        labels = []
        for i in range(N):
            label = int(rows[i][0])
            pixels = list(map(int, rows[i][1:]))
            if len(pixels) != 784:
                raise ValueError("第 {} 行像素数不是 784".format(i))
            labels.append(label)
            arrname = f"sample_{i}"
            f.write(f"static const uint8_t {arrname}[784] = {{\n")
            # 美化一下排版，每行 28 个数字
            for r in range(28):
                line = ", ".join(str(pixels[r*28 + c]) for c in range(28))
                f.write("  " + line + ("," if r < 27 else "") + "\n")
            f.write("};\n\n")
            f.write(f"static const int sample_{i}_label = {label};\n\n")
        # 索引数组
        f.write("static const uint8_t* const kSamples[{}] = {{ {} }};\n".format(
            N, ", ".join([f"sample_{i}" for i in range(N)])
        ))
        f.write("static const int kSampleLabels[{}] = {{ {} }};\n".format(
            N, ", ".join(map(str, labels))
        ))
        f.write("static const int kNumSamples = {};\n".format(N))
    print(f"Generated: {OUT_HDR}  (包含 {N} 张真实 MNIST 样例和其标签)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
