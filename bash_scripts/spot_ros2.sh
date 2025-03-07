source /opt/ros/humble/setup.sh
mkdir -p ros2_ws/src
cd ros2_ws/src
git clone https://github.com/bdaiinstitute/spot_ros2.git

cd spot_ros2
git submodule init
git submodule update

./install_spot_ros2.sh

cd ../../
colcon build --symlink-install

source install/setup.bash
cd ../