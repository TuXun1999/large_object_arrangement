# Use official Ubuntu 22.04 base image
# FROM ubuntu:22.04
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

#RMW ZENOH experimental flag
ARG EXPERIMENTAL_ZENOH_RMW=FALSE

# Set noninteractive mode for APT
ENV DEBIAN_FRONTEND=noninteractive

# Env setup
ENV SHELL=/bin/bash
SHELL ["/bin/bash", "-c"]

# Install system dependencies
RUN apt-get update && apt-get install -y \
    locales \
    curl \
    git \
    && locale-gen en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# Add ROS 2 apt repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update

# Install dependencies
RUN apt-get update -q && \
    apt-get install -yq --no-install-recommends \
    wget \ 
    software-properties-common \ 
    python3-pip \
    python-is-python3 \
    python3-argcomplete \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-rosdep \
    libpython3-dev \
    python3-tk \
    ros-humble-ros-base \
    ros-dev-tools \
    #check if Zenoh should be installed
    $(if [ "$EXPERIMENTAL_ZENOH_RMW" = "TRUE" ]; then echo "ros-humble-rmw-zenoh-cpp"; fi) \
    && rm -rf /var/lib/apt/lists/*





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
