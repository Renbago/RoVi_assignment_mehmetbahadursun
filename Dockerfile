FROM ubuntu:22.04

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && apt-get install -y \
    software-properties-common \
    && apt-get update && apt-get install -y \  
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential cmake git wget curl \
    # OpenGL/Rendering (works with AMD Mesa, NVIDIA, or CPU for rendering GUI)
    libgl1-mesa-glx libgl1-mesa-dev libgl1-mesa-dri \
    libosmesa6 libosmesa6-dev \
    libglfw3 libglfw3-dev libglew-dev \
    libglu1-mesa-dev freeglut3-dev mesa-common-dev mesa-utils \
    libegl1-mesa libegl1-mesa-dev \
    # X11 for GUI
    xauth \
    # OpenCV & Open3D deps
    libopencv-dev libglib2.0-0 libsm6 libxext6 libxrender-dev libfontconfig1 \
    libusb-1.0-0 libjpeg-turbo8 \
    # OMPL deps
    libboost-all-dev libode-dev libeigen3-dev libccd-dev libfcl-dev \
    # some needed tools
    poppler-utils patchelf unzip sudo nano bash-completion \
    && rm -rf /var/lib/apt/lists/*

# User settings 
ENV USER=robotics
RUN useradd -m -s /bin/bash -G sudo ${USER} && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    usermod -aG sudo ${USER} && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER}

USER ${USER}
WORKDIR /home/${USER}

# Install dependencies + tools for mujoco + ompl
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    mujoco roboticstoolbox-python spatialmath-python \
    matplotlib pillow opencv-python opencv-python-headless \
    open3d pandas pdf2image scipy pyyaml glfw ompl mjc_viewer && \
    pip install --no-cache-dir "numpy<2.0" --force-reinstall 

ENV PATH="/home/${USER}/.local/bin:$PATH"

WORKDIR /home/${USER}/workspace

COPY --chown=${USER}:${USER} . /home/${USER}/workspace/

ENV QT_X11_NO_MITSHM=1

WORKDIR /home/${USER}/workspace/robotics
CMD ["/bin/bash"]