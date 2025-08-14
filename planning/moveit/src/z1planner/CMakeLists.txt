cmake_minimum_required(VERSION 3.8)
project(z1planner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

add_executable(z1planner src/planning.cpp)
# add_executable(z1planner src/grasp.cpp)

ament_target_dependencies(z1planner
  rclcpp
  geometry_msgs
  moveit_ros_planning_interface
  tf2_geometry_msgs
)

install(TARGETS z1planner
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
