mkdir --parents  ~/.gazebo/models

# copy model files
cp -r ~/catkin_ws/src/hw3/helpers/models/models/* ~/.gazebo/models

# copy basket mesh files
cp -r ~/catkin_ws/src/hw3/helpers/models/turtlebot/basket ~/catkin_ws/src/turtlebot3/turtlebot3_description/meshes/

# copy xacro file
cp -r ~/catkin_ws/src/hw3/helpers/models/turtlebot/turtlebot3_waffle.urdf.xacro ~/catkin_ws/src/turtlebot3/turtlebot3_description/urdf/

# catkin make
cd ~/catkin_ws && catkin_make

# source latest
source ~/catkin_ws/devel/setup.bash