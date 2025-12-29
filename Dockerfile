# 1. Use CUDA 13 Devel image (includes nvcc and cuDNN headers)
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

# 2. Install Build Tools & ARM-compatible Math Libraries
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    libopenblas-dev \
    iproute2 \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 3. Clone CTranslate2 (Recursive for submodules)
WORKDIR /tmp
RUN git clone --recursive https://github.com/OpenNMT/CTranslate2.git

# 4. Build C++ Backend (Optimized for Blackwell/GB10 via PTX)
WORKDIR /tmp/CTranslate2/build
RUN cmake .. \
    -DWITH_MKL=OFF \
    -DWITH_OPENBLAS=ON \
    -DWITH_CUDA=ON \
    -DWITH_CUDNN=ON \
    -DOPENMP_RUNTIME=COMP \
    -DCUDA_ARCH_NAME=Manual \
    -DCUDA_ARCH_LIST="9.0+PTX" \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    && make -j$(nproc) \
    && make install \
    && ldconfig

# 5. Build Python Bindings
WORKDIR /tmp/CTranslate2/python
RUN pip3 install --break-system-packages -r install_requirements.txt
RUN python3 setup.py bdist_wheel
RUN pip3 install --break-system-packages dist/*.whl
RUN pip3 install --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/cu130
# 6. Install Dependencies (Faster Whisper, etc.)
WORKDIR /workspace
RUN pip3 install --break-system-packages faster-whisper
RUN pip3 install --break-system-packages accelerate
# RUN pip3 install --break-system-packages dotenv
# 7. Runtime Env (Ensure system sees the new libs)
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

RUN pip3 install --break-system-packages spacy

# 2. CRITICAL FIX: Pre-download the Spacy model for Kokoro/Misaki
#    We use --break-system-packages here so the build succeeds.
#    Kokoro usually requires 'en_core_web_md'.
RUN python3 -m spacy download en_core_web_md --break-system-packages
RUN python3 -m spacy download en_core_web_sm --break-system-packages

# 8. Application Setup
#    Change Workdir to match your project structure
WORKDIR /app

#    A. Copy Requirements and install (Filtering out ctranslate2)
COPY voice-sandwich-demo/components/python/requirements.txt .
RUN grep -v "ctranslate2" requirements.txt > req_safe.txt && \
    pip3 install --break-system-packages -r req_safe.txt

#    B. Copy Source Code (REQUIRED so the container has the code)
COPY . .
ENV PYTHONPATH="/app/voice-sandwich-demo/components/python/src:/app/VibeVoice:/app"

WORKDIR /app/voice-sandwich-demo/components/python
CMD ["python3", "src/main.py"]

















# # 1. Use CUDA 13 Devel image (with cuDNN headers)
# FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

# # 2. Install Build Tools & ARM-compatible Math Libraries
# # 2. Install Build Tools & System Dependencies
# # 2. Install system dependencies
# # REMOVED: python3-distutils (It does not exist in Ubuntu 24.04)
# RUN apt-get update && apt-get install -y \
#     git \
#     cmake \
#     build-essential \
#     python3 \
#     python3-dev \
#     python3-pip \
#     python3-venv \
#     libopenblas-dev \
#     iproute2 \
#     wget \
#     curl \
#     ca-certificates \
#     && rm -rf /var/lib/apt/lists/*

# # Your next step handles the replacement (setuptools)
# RUN curl -LsSf https://astral.sh/uv/install.sh | sh


# # 3. Clone CTranslate2
# WORKDIR /tmp
# RUN git clone --recursive https://github.com/OpenNMT/CTranslate2.git

# # 4. Build C++ Backend
# #    - DOPENMP_RUNTIME=COMP: Fixes Intel/ARM crash
# #    - DCUDA_ARCH_LIST="9.0+PTX": Targets Hopper + Future Compat (Blackwell)
# WORKDIR /tmp/CTranslate2/build
# RUN cmake .. \
#     -DWITH_MKL=OFF \
#     -DWITH_OPENBLAS=ON \
#     -DWITH_CUDA=ON \
#     -DWITH_CUDNN=ON \
#     -DOPENMP_RUNTIME=COMP \
#     -DCUDA_ARCH_NAME=Manual \
#     -DCUDA_ARCH_LIST="9.0+PTX" \
#     -DCMAKE_INSTALL_PREFIX=/usr/local \
#     && make -j$(nproc) \
#     && make install \
#     && ldconfig

# # 5. Build Python Bindings
# WORKDIR /tmp/CTranslate2/python
# RUN pip3 install --break-system-packages -r install_requirements.txt
# RUN python3 setup.py bdist_wheel
# RUN pip3 install --break-system-packages dist/*.whl

# # 6. Install Dependencies
# WORKDIR /workspace
# RUN pip3 install --break-system-packages faster-whisper

# # 7. Runtime Env
# ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# WORKDIR /voice-sandwich-demo/components/python/
# COPY requirements.txt .
# RUN grep -v "ctranslate2" requirements.txt > req_safe.txt && \
#     pip3 install --break-system-packages -r req_safe.txt

# # 3. Then run your app
# CMD ["python3", "src/main.py"]
