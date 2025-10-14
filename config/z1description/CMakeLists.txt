cmake_minimum_required(VERSION 3.8)
project(z1description)


find_package(ament_cmake REQUIRED)
find_package(xacro REQUIRED)

install(DIRECTORY urdf
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY meshes
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
