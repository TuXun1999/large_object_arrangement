CALL_DIR=$PWD
# Import several helper functions in bash
SCRIPT_DIR=$PWD
. "${SCRIPT_DIR}/tools.sh"

# Path to Spot workspace, relative to repository root;
# No begin or trailing slash.
SPOT_PATH=${SCRIPT_DIR}

# Add a few alias for pinging spot.
#------------- Main Logic  ----------------

# We have only tested Spot stack with Ubuntu 22.04.
if ! ubuntu_version_equal 22.04; then
    echo "Current SPOT development requires Ubuntu 22.04. Abort."
    return 1
fi


# create a dedicated virtualenv for all programs
if [ ! -d "${SPOT_PATH}/venv/spot" ]; then
    cd ${SPOT_PATH}/
    virtualenv -p python3 venv/spot
fi

# activate virtualenv; 
source ${SPOT_PATH}/venv/spot/bin/activate
pip install -U pip>=20.3
## Install the dependencies
pip install empy==3.3.4
sudo apt-get install '^libxcb.*-dev' libx11-xcb-dev xorg-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
sudo apt-get install libsm6 libxrender1 libfontconfig1

# pip3 install torch torchvision torchaudio
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy
pip install open3d==0.18.0
pip install trimesh
pip install pyrender

# scikit-image
# pip install scipy==1.10.0
pip install -U scikit-image==0.21.0
pip install mesh2sdf
pip install pydot
pip install graphviz

## Dependencies for SPOT
pip install PySide2
pip install bosdyn-client
pip install bosdyn-mission
pip install bosdyn-api
pip install bosdyn-core
pip install bosdyn-choreography-client






    



