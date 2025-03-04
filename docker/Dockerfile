FROM osrf/ros:humble-desktop-full

# Install dependencies (taken from devcontainer dockerfile)
RUN DEBIAN_FRONTEND=noninteractive apt-get update -q && \
    apt-get update -q && \
    apt-get install -yq --no-install-recommends \
    curl \
    wget \
    python3-pip \
    python-is-python3 \
    python3-argcomplete \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-rosdep \
    libpython3-dev && \
    rm -rf /var/lib/apt/lists/*

# Create ROS workspace
WORKDIR /ros_ws/src

# Clone driver code
RUN git clone https://github.com/bdaiinstitute/spot_ros2.git .
RUN git submodule update --init --recursive

# Run install script
RUN /ros_ws/src/install_spot_ros2.sh

# Build packages with Colcon
WORKDIR /ros_ws/
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install --packages-ignore proto2ros proto2ros_tests

# create a user
ARG hostuser
ARG hostgroup
ARG hostuid
ARG hostgid

RUN echo Host user is $hostuser:$hostuser
RUN groupadd --gid $hostgid $hostgroup
RUN adduser --disabled-password --gecos '' --gid $hostgid --uid $hostuid $hostuser
RUN adduser $hostuser sudo
# Ensure sudo group users are not asked for a password when using sudo command
# by ammending sudoers file
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> \
/etc/sudoers

USER $hostuser
WORKDIR /home/$hostuser
ENV HOME=/home/$hostuser

# Different shell color
RUN echo "export PS1='\[\033[01;31m\]\u@\h\[\033[00m\]:\[\033[01;33m\]\w\[\033[00m\]$ '" >> $HOME/.bashrc

# print some info on start
RUN echo "echo -e 'Welcome! You are now in a docker container ().'" >> $HOME/.bashrc
RUN echo "echo -e \"Docker ID: $(basename $(cat /proc/1/cpuset))\"" >> $HOME/.bashrc
CMD ["bash"]
