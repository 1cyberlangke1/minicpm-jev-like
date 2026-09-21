#!/bin/sh
# 源码编译 llama.cpp 的 CUDA 后端, 并换装进 llama-cpp-python 的 lib 目录。
#
# 为什么需要它: PyPI 上的 llama-cpp-python wheel 是纯 CPU 版; 官方预编译 CUDA wheel
# 在部分机器上会因指令集过新崩 0xc000001d。自己编最稳。
#
# 输入: 本仓库根目录 (脚本位置自动推导) + VS2022 + CUDA Toolkit 12.x + cmake/ninja
#      (cmake/ninja 可以直接 pip install 到 .venv, 脚本优先用 .venv/Scripts 里的)
# 输出: .venv/.../llama_cpp/lib/ 换成 MSVC+CUDA 构建的 DLL 组, import 后 gpu offload 为 True
# 预期: clone + checkout 与 llama-cpp-python 0.3.35 对应的上游 pin -> CMake Ninja
#      (GGML_CUDA, 计算能力按 $CMAKE_CUDA_ARCHITECTURES, 默认 89) -> 增量编译 ->
#      清旧 DLL -> 拷新 DLL (llama.dll 改名 libllama.dll) -> 冒烟测速。
#      中断后重跑复用 build 目录, ninja 会增量续编。
#
# 用法 (Git Bash): bash scripts/build_llama_cuda.sh
set -e

REPO=$(cd "$(dirname "$0")/.." && pwd)
PIN=4df29be4f4c3673f428170fda944a5b19f743bb8
ARCH=${CMAKE_CUDA_ARCHITECTURES:-89}

case "$REPO" in
    /[a-zA-Z]/*) DRIVE=$(echo "$REPO" | cut -c2 | tr 'A-Z' 'a-z') ;;
    *) echo "只支持盘符路径 (Git Bash 的 /d/... 形式), 收到 $REPO" >&2; exit 1 ;;
esac
REPO_SH=$(echo "$REPO" | sed "s|^/[a-zA-Z]/|/$DRIVE/|")
cd "$REPO_SH"

if [ ! -d tmp/llama.cpp/.git ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp tmp/llama.cpp
fi
cd tmp/llama.cpp
git fetch --depth 1 origin "$PIN"
git checkout -q FETCH_HEAD

# 注入 MSVC 环境 (vcvars64)
VCVARS="C:\\\\Program Files\\\\Microsoft Visual Studio\\\\2022\\\\Community\\\\VC\\\\Auxiliary\\\\Build\\\\vcvars64.bat"
tmpenv=$(mktemp)
cmd.exe //c "call \"$VCVARS\" >nul && set" 2>/dev/null | tr -d '\r' > "$tmpenv" || true
while IFS='=' read -r name value; do
    case "$name" in [A-Za-z_]*) export "$name=$value" ;; esac
done < "$tmpenv"
rm -f "$tmpenv"
export DISTUTILS_USE_SDK=1

# 剔除 msys64/mingw: 它们的 GNU windres 会被 nvcc 抢去编译资源而崩
PATH_NO_MSYS=$(echo "$PATH" | tr ':' '\n' | grep -viE "msys|mingw" | paste -sd:)
export PATH="$REPO_SH/.venv/Scripts:$PATH_NO_MSYS"

cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
      -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF
cmake --build build -j 8

# 换装: 清旧 MinGW 组, 灌新 MSVC+CUDA 组
LIBDIR="$REPO_SH/.venv/Lib/site-packages/llama_cpp/lib"
rm -f "$LIBDIR"/*.dll "$LIBDIR"/*.a
cp build/bin/*.dll "$LIBDIR"/
mv "$LIBDIR"/llama.dll "$LIBDIR"/libllama.dll 2>/dev/null || true
ls "$LIBDIR"

cd "$REPO_SH"
.venv/Scripts/python.exe - <<'EOF'
import time
import llama_cpp
from llama_cpp import llama_supports_gpu_offload

print("gpu offload:", llama_supports_gpu_offload())
model = llama_cpp.Llama(
    model_path="weights/MiniCPM5-2B-Q4_K_M.gguf", n_gpu_layers=-1, n_ctx=512, verbose=False
)
start = time.time()
result = model("Hello", max_tokens=16)
count = result["usage"]["completion_tokens"]
print(f"GPU 全层: {count / (time.time() - start):.1f} tok/s")
EOF
