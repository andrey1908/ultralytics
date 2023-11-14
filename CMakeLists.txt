cmake_minimum_required(VERSION 3.0.2)
project(ultralytics)

find_package(catkin REQUIRED COMPONENTS)

catkin_python_setup()

catkin_package()

# Install package.xml for catkin (does not work because of catkin_python_setup() for some reason)
install(FILES package.xml DESTINATION share/ultralytics/)

