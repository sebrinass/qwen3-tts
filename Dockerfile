# syntax=docker/dockerfile:1

# ============================================================================
# Build stage: compile the tts-server target with the Vulkan backend
# ============================================================================
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        git \
        pkg-config \
        python3 \
        libvulkan-dev \
        spirv-tools \
        glslc \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 22.04's spirv-headers apt package ships only headers, not the
# SPIRV-HeadersConfig.cmake that ggml-vulkan's find_package(SPIRV-Headers
# CONFIG REQUIRED) needs. Install from source so cmake can locate it.
RUN git clone --depth 1 https://github.com/KhronosGroup/SPIRV-Headers.git /tmp/spirv-headers \
    && cmake -S /tmp/spirv-headers -B /tmp/spirv-headers/build \
    && cmake --install /tmp/spirv-headers/build \
    && rm -rf /tmp/spirv-headers

# Clone the project and initialize the ggml submodule. The build adds ggml
# as a CMake subdirectory, so the submodule must be present. --depth 1 is
# enough: version.cmake only reads the HEAD commit hash.
ARG QWENTTS_REPO=https://github.com/ServeurpersoCom/qwentts.cpp.git
RUN git clone --depth 1 "${QWENTTS_REPO}" /src \
    && cd /src \
    && git submodule update --init --recursive --depth 1

# Overlay our modified tts-server.cpp + shared tts-server.h with voice cloning,
# persistence support, and the wav-on-default response_format for browser-compatible
# audio out of the box.
COPY tools/tts-server.cpp /src/tools/tts-server.cpp
COPY src/tts-server.h /src/src/tts-server.h

WORKDIR /src

# Vulkan build, equivalent to buildvulkan.sh. Only the tts-server target is
# built; its dependencies (ggml + ggml-vulkan, qwen-core, cpp-httplib,
# yyjson) are pulled in automatically by CMake.
RUN cmake -B build -DGGML_VULKAN=ON \
    && cmake --build build --config Release -j "$(nproc)" --target tts-server

# Stage a clean /out holding the binary plus any backend shared objects ggml
# may emit in DL mode. ggml_backend_load_all() searches for backends next
# to the executable, so they must ship in the same directory at runtime.
#
# Libraries live in subdirectories like build/ggml/src/ggml-cpu/,
# build/ggml/src/ggml-vulkan/, build/ggml/src/ggml-base/ etc., so we walk
# the full tree. `cp -P` preserves symlinks so the SONAME aliases
# (libggml.so.0 -> libggml.so.0.15.2) survive intact - otherwise the
# dynamic linker cannot find libggml.so.0 at runtime.
RUN mkdir -p /out \
    && cp -P build/tts-server /out/ \
    && find build \( -type f -o -type l \) \( -name "*.so" -o -name "*.so.*" \) \
        -exec sh -c 'dst="/out/$(basename "$1")"; rm -f "$dst"; cp -P "$1" "$dst"' _ {} \;

# ============================================================================
# Runtime stage
# ============================================================================
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Runtime dependencies. tts-server is built with OpenMP enabled (-fopenmp)
# and the ggml backends rely on dynamic linking, so the runtime image needs:
#   libgomp1            - GCC OpenMP runtime (libgomp.so.1)
#   libvulkan1          - Vulkan loader
#   mesa-vulkan-drivers - Intel (ANV) + AMD (RADV) Vulkan ICDs (the actual
#                         Intel iGPU Vulkan driver used for compute)
#   vulkan-tools        - vulkaninfo for debugging (optional)
#   intel-media-va-driver - Intel iGPU media (VAAPI) support
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libvulkan1 \
        mesa-vulkan-drivers \
        vulkan-tools \
        intel-media-va-driver \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Binary + backend .so from the build stage. Same directory so ggml can
# dlopen the Vulkan backend at runtime.
COPY --from=builder /out/ /app/

RUN mkdir -p /app/models /app/voices

# Default configuration. Models are NOT baked into the image; mount them
# under /app/models (see docker-compose.yml). Override at runtime via env.
ENV TALKER_MODEL=/app/models/talker.gguf \
    CODEC_MODEL=/app/models/codec.gguf \
    VOICES_DIR=/app/voices \
    PORT=8080 \
    LD_LIBRARY_PATH=/app

EXPOSE 8080

# --host 0.0.0.0 is mandatory inside a container so the published port is
# reachable (the binary defaults to 127.0.0.1, which is not). --lang auto
# mirrors examples/server.sh.
ENTRYPOINT ["sh","-c","exec ./tts-server --model \"$TALKER_MODEL\" --codec \"$CODEC_MODEL\" --host 0.0.0.0 --port \"$PORT\" --lang auto"]
