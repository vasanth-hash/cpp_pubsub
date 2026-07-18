cmake_minimum_required(VERSION 3.5)
project(cpp_pubsub)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(mavros_msgs)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(mavros_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(autoware_control_msgs REQUIRED)
find_package(autoware_vehicle_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

add_executable(drive src/trov_drive_bridge.cpp)
ament_target_dependencies(drive 
  rclcpp 
  std_msgs 
  mavros_msgs 
  geometry_msgs 
  nav_msgs
  tf2
  tf2_ros
  sensor_msgs
  visualization_msgs
  )
install(TARGETS
 drive
 DESTINATION lib/${PROJECT_NAME})

add_executable(drive_outdoor src/trov_drive_bridge_outdoor.cpp)
ament_target_dependencies(drive_outdoor 
  rclcpp 
  std_msgs 
  mavros_msgs 
  geometry_msgs 
  nav_msgs
  visualization_msgs
  tf2
  tf2_ros
  )

install(TARGETS
 drive_outdoor
 DESTINATION lib/${PROJECT_NAME})

add_executable(trov_teleop src/trov_teleop.cpp)
ament_target_dependencies(trov_teleop 
  rclcpp 
  geometry_msgs
  )
install(TARGETS
 trov_teleop
 DESTINATION lib/${PROJECT_NAME})

add_executable(line_follower_bridge src/line_follower_bridge.cpp)
ament_target_dependencies(line_follower_bridge 
  rclcpp 
  geometry_msgs
  mavros_msgs
  )
install(TARGETS
 line_follower_bridge
 DESTINATION lib/${PROJECT_NAME})

add_executable(trov_autoware_bridge src/trov_autoware_bridge.cpp)
ament_target_dependencies(trov_autoware_bridge 
  rclcpp 
  geometry_msgs
  mavros_msgs
  nav_msgs
  tf2_ros
  tf2_geometry_msgs
  autoware_control_msgs
  autoware_vehicle_msgs
  )
install(TARGETS
 trov_autoware_bridge
 DESTINATION lib/${PROJECT_NAME})

ament_package()
