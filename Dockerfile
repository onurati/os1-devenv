ARG PYTHON_IMAGE=python:3.10-slim-bookworm
ARG OUSTER_SDK_REF=v0.16.1
ARG ROSBAGS_VERSION=0.11.0


FROM ${PYTHON_IMAGE} AS builder

ARG OUSTER_SDK_REF
ARG ROSBAGS_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CC=gcc \
    CXX=g++

RUN apt-get -o Acquire::Retries=3 update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        pkg-config \
        flatbuffers-compiler \
        libblas-dev \
        libceres-dev \
        libcurl4-openssl-dev \
        libeigen3-dev \
        libflatbuffers-dev \
        libgflags-dev \
        libglfw3-dev \
        libjpeg62-turbo-dev \
        libjsoncpp-dev \
        liblapack-dev \
        libpcap-dev \
        libpng-dev \
        libssl-dev \
        libsuitesparse-dev \
        libtbb-dev \
        libtiff-dev \
        libtins-dev \
        libudev-dev \
        libzip-dev \
        libzstd-dev \
        zlib1g-dev \
    && python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install --no-cache-dir \
        ezdxf \
        matplotlib \
        more-itertools \
        numpy \
        open3d \
        packaging \
        "Pillow>=10.2.0" \
        "prettytable>=2.1.0,<3.17.0" \
        "psutil>=5.9.5,<6" \
        requests \
        shapely \
        "click>=8.1.3,<8.2.0" \
        flask==3.0 \
        "laspy>=2.5.0,<3" \
        "rosbags==${ROSBAGS_VERSION}" \
        threadpoolctl \
        waitress==3.0.0 \
        "zeroconf>=0.131.0" \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/venv/bin:${PATH}

RUN git clone --depth 1 https://github.com/Tessil/robin-map.git /tmp/robin-map \
    && cmake -S /tmp/robin-map -B /tmp/robin-map/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_STANDARD=20 \
        -DCMAKE_CXX_STANDARD_REQUIRED=ON \
        -DROBIN_MAP_BUILD_TESTS=OFF \
    && cmake --build /tmp/robin-map/build -j"$(nproc)" \
    && cmake --install /tmp/robin-map/build \
    && git clone --depth 1 --branch ${OUSTER_SDK_REF} https://github.com/ouster-lidar/ouster-sdk.git /tmp/ouster-sdk \
    && python -c "from pathlib import Path; path = Path('/tmp/ouster-sdk/CMakeLists.txt'); content = path.read_text(); path.write_text('set(CMAKE_CXX_STANDARD 20)\\nset(CMAKE_CXX_STANDARD_REQUIRED ON)\\n' + content)" \
    && cd /tmp/ouster-sdk/python \
    && CMAKE_ARGS="-DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_STANDARD_REQUIRED=ON" \
       CXXFLAGS="-std=c++20" \
       CFLAGS="-std=c99" \
         pip install --no-cache-dir --no-deps . \
     && pip install --no-cache-dir --upgrade "rosbags==${ROSBAGS_VERSION}" \
    && python -c "from pathlib import Path; path = Path('/opt/venv/lib/python3.10/site-packages/open3d/__init__.py'); text = path.read_text(); text = text.replace('import open3d.visualization\\n', 'try:\\n    import open3d.visualization\\nexcept ImportError:\\n    pass\\n'); text = text.replace('import open3d.ml\\n', 'try:\\n    import open3d.ml\\nexcept ImportError:\\n    pass\\n'); path.write_text(text)" \
    && rm -rf \
        /opt/venv/lib/python3.10/site-packages/open3d/cuda \
        /opt/venv/lib/python3.10/site-packages/open3d/ml \
        /opt/venv/lib/python3.10/site-packages/open3d/_ml3d \
        /opt/venv/lib/python3.10/site-packages/open3d/examples \
        /opt/venv/lib/python3.10/site-packages/open3d/labextension \
        /opt/venv/lib/python3.10/site-packages/open3d/nbextension \
    && pip uninstall -y dash plotly pandas scikit-learn scipy \
    && rm -rf /tmp/robin-map /tmp/ouster-sdk \
    && find /opt/venv -type d -name __pycache__ -prune -exec rm -rf {} + \
    && find /opt/venv -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete \
    && find /opt/venv -type f -name '*.a' -delete


FROM ${PYTHON_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OUSTER_BAG_DIR=/data/ouster_bags \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH}

RUN apt-get -o Acquire::Retries=3 update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libatlas3-base \
        libcholmod3 \
        libcurl4 \
        libcxsparse3 \
        libgflags2.2 \
        libgl1 \
        libglfw3 \
        libglib2.0-0 \
        libgoogle-glog0v6 \
        libgomp1 \
        libjpeg62-turbo \
        libjsoncpp25 \
        liblapack3 \
        libpcap0.8 \
        libpng16-16 \
        libssl3 \
        libsm6 \
        libsuitesparseconfig5 \
        libtbb12 \
        libtiff6 \
        libtins4.0 \
        libudev1 \
        libusb-1.0-0 \
        libxext6 \
        libxrender1 \
        libzip4 \
        libzstd1 \
    && mkdir -p ${OUSTER_BAG_DIR} \
    && printf '%s\n' 'export VIRTUAL_ENV=/opt/venv' 'export PATH=/opt/venv/bin:$PATH' > /etc/profile.d/venv.sh \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /workspace

VOLUME ["/data/ouster_bags"]
CMD ["/bin/bash"]