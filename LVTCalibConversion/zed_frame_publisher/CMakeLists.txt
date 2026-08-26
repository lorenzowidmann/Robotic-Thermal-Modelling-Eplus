cmake_minimum_required(VERSION 3.0.2)
project(zed_frame_publisher)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  cv_bridge
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/zed_frame_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
