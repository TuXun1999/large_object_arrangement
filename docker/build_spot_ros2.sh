# UID/GID for the container user
hostuser=$USER
hostuid=$UID
hostgroup=$(id -gn $hostuser)
hostgid=$(id -g $hostuser)

docker build \
       -t spot_ros2_image:latest\
       --build-arg hostuser=$hostuser\
       --build-arg hostgroup=$hostgroup\
       --build-arg hostuid=$hostuid\
       --build-arg hostgid=$hostgid\
       --rm\
       .