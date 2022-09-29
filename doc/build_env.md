**Build Environment**

[deepstream_python_apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)

[deepstream_python_apps - howto](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/blob/master/HOWTO.md )

[DeepStream python bindings](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/bindings)

```shell
# Run Container
docker run -it --gpus=all --name=bs-face --net=host -it -v /tmp/.X11-unix/:/tmp/.X11-unix --env-file  .devcontainer/devcontainer.env nvcr.io/nvidia/deepstream:6.1-devel bash

# .devcontainer/devcontainer.env
DISPLAY=:0

# s - Base dependencies
apt install python3-gi python3-dev python3-gst-1.0 python-gi-dev git python-dev \
    python3 python3-pip python3.8-dev cmake g++ build-essential libglib2.0-dev \
    libglib2.0-dev-bin libgstreamer1.0-dev libtool m4 autoconf automake libgirepository1.0-dev libcairo2-dev
    
# s - Initialization of submodules
<DeepStream 6.1 ROOT>/sources/deepstream_python_apps
cd /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/
git submodule update --init

# s - Installing Gst-python
sudo apt-get install -y apt-transport-https ca-certificates -y
sudo update-ca-certificates

cd 3rdparty/gst-python/
./autogen.sh
make
sudo make install

# s - Compiling the bindings
cd deepstream_python_apps/bindings
mkdir build
cd build
cmake ..
make

# s - Installing the bindings
# Installing the pip wheel
pip3 install ./pyds-1.1.3-py3-none*.whl

# pip wheel troubleshooting
# If the wheel installation fails, upgrade the pip using the following command
python3 -m pip install --upgrade pip

# s - Launching test-1 app
cd apps/deepstream-test1
python3 deepstream_test_1.py <input .h264 file> 

```