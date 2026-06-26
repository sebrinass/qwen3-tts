# syntax=docker/dockerfile:1

# ============================================================================
# Build stage: compile the tts-server target with the Vulkan backend
# ============================================================================
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        git \
        pkg-config \
        python3 \
        libvulkan-dev \
        vulkan-headers \
    && rm -rf /var/lib/apt/lists/*

# Clone the project and initialize the ggml submodule. The build adds ggml
# as a CMake subdirectory, so the submodule must be present. --depth 1 is
# enough: version.cmake only reads the HEAD commit hash.
ARG QWENTTS_REPO=https://github.com/ServeurpersoCom/qwentts.cpp.git
RUN git clone --depth 1 "${QWENTTS_REPO}" /src \
    && cd /src \
    && git submodule update --init --recursive --depth 1

# Overlay our modified tts-server.cpp with voice cloning + persistence support
COPY tools/tts-server.cpp /src/tools/tts-server.cpp

WORKDIR /src

# Vulkan build, equivalent to buildvulkan.sh. Only the tts-server target is
# built; its dependencies (ggml + ggml-vulkan, qwen-core, cpp-httplib,
# yyjson) are pulled in automatically by CMake.
RUN cmake -B build -DGGML_VULKAN=ON \
    && cmake --build build --config Release -j "$(nproc)" --target tts-server

# Stage a clean /out holding the binary plus any backend shared objects ggml
# may emit in DL mode. ggml_backend_load_all() searches for backends next
# to the executable, so they must ship in the same directory at runtime.
RUN mkdir -p /out \
    && cp build/tts-server /out/ \
    && find build -maxdepth 1 -type f -name "*.so*" -exec cp -L {} /out/ \;

# ============================================================================
# Runtime stage
# ============================================================================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Runtime Vulkan stack:
#   libvulkan1           - Vulkan loader
#   mesa-vulkan-drivers  - Intel (ANV) + AMD (RADV) Vulkan ICDs (this is the
#                          actual Intel iGPU Vulkan driver used for compute)
#   vulkan-tools         - vulkaninfo for debugging (optional)
#   intel-media-va-driver- Intel iGPU media (VAAPI) support
RUN apt-get update && apt-get install -y --no-install-recommends \
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
