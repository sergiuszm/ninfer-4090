# syntax=docker/dockerfile:1

FROM nvidia/cuda:13.1.2-devel-ubuntu24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        cmake \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libcurl4-openssl-dev \
        libhwloc-dev \
        libjpeg-dev \
        libswscale-dev \
        liburing-dev \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Link against the toolkit stub when the build stage has no host GPU driver.
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1

WORKDIR /src
COPY . .

RUN cmake -S . -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DNINFER_TARGET=rtx5090 \
        -DNINFER_BUILD_APPS=ON \
        -DBUILD_TESTING=OFF \
        -DNINFER_BUILD_BENCHMARKS=OFF \
    && cmake --build /build --parallel --target ninfer ninfer-serve

FROM nvidia/cuda:13.1.2-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libavcodec60 \
        libavformat60 \
        libavutil58 \
        libcurl4t64 \
        libhwloc15 \
        libjpeg-turbo8 \
        libswscale7 \
        liburing2 \
    && rm -rf /var/lib/apt/lists/*

# The CUDA runtime image ships forward-compatibility libraries in
# /usr/local/cuda*/compat (a newer libcuda.so than the host driver). Forward
# compatibility is supported only on datacenter GPUs; on any GeForce card the
# loader picks these up and every CUDA call fails at startup with
#   cudaErrorCompatNotSupportedOnDevice: forward compatibility was attempted
#   on non supported HW
# Removing them lets the container use the host driver through ordinary CUDA
# minor-version compatibility, which is what an RTX 3090/3090 Ti needs.
RUN rm -rf /usr/local/cuda-13.1/compat /usr/local/cuda-13/compat /usr/local/cuda/compat

COPY --from=build /build/apps/ninfer /usr/local/bin/ninfer
COPY --from=build /build/apps/ninfer-serve /usr/local/bin/ninfer-serve
COPY --from=build /build/lib/ /usr/local/lib/

WORKDIR /workspace
EXPOSE 8080
STOPSIGNAL SIGTERM

CMD ["ninfer-serve", "--help"]
