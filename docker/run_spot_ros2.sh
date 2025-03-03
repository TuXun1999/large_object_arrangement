    docker run -it\
           --env "TERM=xterm-256color"\
           --env "DISPLAY=$DISPLAY"\
           --volume /tmp/.X11-unix/:/tmp/.X11-unix:rw\
           --env "XAUTHORITY=$XAUTH"\
           --volume $XAUTH:$XAUTH\
           --privileged\
           --network=host\
           --name="spot_ros2_dev"\
           --gpus all\
           --ipc=host\
           spot_ros2_image:latest
           
