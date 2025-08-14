#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Geometry>
#include <chrono>
#include <cmath>

using namespace std::chrono_literals;

class SimPosePlanner : public rclcpp::Node
{
public:
  SimPosePlanner(const moveit::planning_interface::MoveGroupInterfacePtr &mgi1,
                 const moveit::planning_interface::MoveGroupInterfacePtr &mgi2)
      : Node("sim_pose_planner"),
        move_group_interface_(mgi1),
        gripper_group_interface_(mgi2),
        last_pose_received_(false),
        seed_published_(false)
  {
    RCLCPP_INFO(this->get_logger(), "Initializing SimPosePlanner...");

    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("chair_target_marker", 10);

    // MoveGroup 配置
    move_group_interface_->setPlanningTime(5.0);
    move_group_interface_->setNumPlanningAttempts(5);
    move_group_interface_->setStartStateToCurrentState();

    gripper_group_interface_->setPlanningTime(2.0);
    gripper_group_interface_->setNumPlanningAttempts(2);
    gripper_group_interface_->setStartStateToCurrentState();

    // 发布规划后关节角（给你自己做可视化/记录用）
    joint_state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("moveit_joint_states", 10);

    // 给 MoveIt 的 CurrentStateMonitor 播一帧 joint_states 种子（只在需要时）
    joint_states_seed_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_states", rclcpp::QoS(10));

    // 先把椅子放进 PlanningScene
    addChairCollisionObject();

    // 使用定时器延迟并反复检查当前状态是否就绪
    timer_ = this->create_wall_timer(300ms, std::bind(&SimPosePlanner::tryPlanOnce, this));

    RCLCPP_INFO(this->get_logger(), "SimPosePlanner is ready, waiting for robot current state...");
  }

private:
  // ========== 关键：等待状态就绪再规划 ==========
  void tryPlanOnce()
  {
    // 先尝试获取当前状态
    moveit::core::RobotStatePtr current_state = move_group_interface_->getCurrentState(0.1); // 带小超时
    if (!current_state)
    {
      // 如果还没有状态，且还没播过种子，则播一次全零关节态帮助 CurrentStateMonitor 初始化
      if (!seed_published_)
      {
        auto names = move_group_interface_->getJointNames();
        if (!names.empty())
        {
          sensor_msgs::msg::JointState js;
          js.header.stamp = this->now();
          js.name = names;
          js.position.resize(names.size(), 0.0);
          joint_states_seed_pub_->publish(js);
          seed_published_ = true;
          RCLCPP_WARN(this->get_logger(),
                      "No current state yet; published a one-time joint_states seed (all zeros).");
        }
        else
        {
          RCLCPP_WARN(this->get_logger(),
                      "MoveGroupInterface returned empty joint names; cannot publish seed yet.");
        }
      }
      // 等下一次 timer 再试
      return;
    }

    // 有状态了，停止定时器，开始规划
    if (timer_)
    {
      timer_->cancel();
      timer_.reset();
    }

    RCLCPP_INFO(this->get_logger(), "Robot current state is available. Starting planToChairTop().");
    planToChairTop();
  }

public:
  void planToChairTop()
  {
    // 再次防御式检查当前状态，避免竞态
    moveit::core::RobotStatePtr current_state = move_group_interface_->getCurrentState(0.1);
    if (!current_state)
    {
      RCLCPP_ERROR(this->get_logger(), "Current state not available; aborting planToChairTop.");
      return;
    }

    openGripper();

    // 椅背参数（与 addChairCollisionObject 对应）
    double back_height = 0.4; // 椅背高度（CollisionObject 定义的 Z 尺寸）

    geometry_msgs::msg::Pose back_pose;
    back_pose.position.x = 0.5 - 0.025; // 靠近座位后方
    back_pose.position.y = 0.0;
    back_pose.position.z = 0.35; // 椅背中心高度（用于计算顶部）
    {
      tf2::Quaternion q;
      q.setRPY(0, -30.0 * M_PI / 180.0, 0); // 椅背后倾 30 度
      back_pose.orientation = tf2::toMsg(q);
    }

    // 将中心点转换为顶部点
    tf2::Vector3 local_top(0.0, 0.0, back_height / 2.0);
    tf2::Quaternion tf2_q;
    tf2::fromMsg(back_pose.orientation, tf2_q);
    tf2::Matrix3x3 rot(tf2_q);
    tf2::Vector3 world_top_offset = rot * local_top;

    tf2::Vector3 top_position(
        back_pose.position.x + world_top_offset.x(),
        back_pose.position.y + world_top_offset.y(),
        back_pose.position.z + world_top_offset.z());

    const Eigen::Isometry3d &ee_tf = current_state->getGlobalLinkTransform("gripperStator");
    const Eigen::Isometry3d &wrist_tf = current_state->getGlobalLinkTransform("link06");
    double ee_to_wrist_dist = (ee_tf.translation() - wrist_tf.translation()).norm();
    RCLCPP_INFO(this->get_logger(), "Distance from gripperStator to link06: %.3f m", ee_to_wrist_dist);

    // 椅背外法线（假设局部 +X 方向是朝外）
    tf2::Vector3 local_normal(1.0, 0.0, 0.0);
    tf2::Vector3 world_normal = rot * local_normal;
    world_normal.normalize();

    // 沿法线反向延长
    tf2::Vector3 target_position = top_position - world_normal * ee_to_wrist_dist;

    // 构造目标位姿
    geometry_msgs::msg::PoseStamped target_pose;
    target_pose.header.frame_id = "world";
    target_pose.header.stamp = this->now();
    target_pose.pose.position.x = target_position.x();
    target_pose.pose.position.y = target_position.y();
    target_pose.pose.position.z = target_position.z();

    // 姿态可以直接使用椅背方向（这里微调 35° 作为示例）
    {
      tf2::Quaternion q;
      q.setRPY(0, 35.0 * M_PI / 180.0, 0);
      target_pose.pose.orientation = tf2::toMsg(q);
    }

    RCLCPP_INFO(this->get_logger(),
                "Planning to offset chair top at (%.3f, %.3f, %.3f), offset=%.3f m",
                target_pose.pose.position.x,
                target_pose.pose.position.y,
                target_pose.pose.position.z,
                ee_to_wrist_dist);

    // 创建 marker
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "world";
    marker.header.stamp = this->now();
    marker.ns = "chair_target";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose = target_pose.pose;
    marker.scale.x = 0.05;
    marker.scale.y = 0.05;
    marker.scale.z = 0.05;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.color.a = 0.8;
    marker_pub_->publish(marker);

    // 调用规划执行
    planAndExecute(std::make_shared<geometry_msgs::msg::PoseStamped>(target_pose));
  }

  void addChairCollisionObject()
  {
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

    // 椅座
    moveit_msgs::msg::CollisionObject seat;
    seat.header.frame_id = "world";
    seat.id = "chair_seat";

    shape_msgs::msg::SolidPrimitive seat_shape;
    seat_shape.type = shape_msgs::msg::SolidPrimitive::BOX;
    seat_shape.dimensions = {0.4, 0.4, 0.02}; // 宽、深、厚度

    geometry_msgs::msg::Pose seat_pose;
    seat_pose.position.x = 0.6;
    seat_pose.position.y = 0.0;
    seat_pose.position.z = 0.1; // 高度一半
    seat_pose.orientation.w = 1.0;

    seat.primitives.push_back(seat_shape);
    seat.primitive_poses.push_back(seat_pose);
    seat.operation = seat.ADD;

    // 椅背（倾斜）
    moveit_msgs::msg::CollisionObject backrest;
    backrest.header.frame_id = "world";
    backrest.id = "chair_backrest";

    shape_msgs::msg::SolidPrimitive back_shape;
    back_shape.type = shape_msgs::msg::SolidPrimitive::BOX;
    back_shape.dimensions = {0.02, 0.4, 0.4}; // 宽、厚、背高

    geometry_msgs::msg::Pose back_pose;
    back_pose.position.x = 0.5 - 0.025; // 靠近座位后方
    back_pose.position.y = 0.0;
    back_pose.position.z = 0.2; // 椅背中心高度（注意：与 planToChairTop 用于计算顶部的 0.35 不同）
    {
      tf2::Quaternion q;
      q.setRPY(0, -30.0 * M_PI / 180.0, 0); // 椅背后倾 30 度
      back_pose.orientation = tf2::toMsg(q);
    }

    backrest.primitives.push_back(back_shape);
    backrest.primitive_poses.push_back(back_pose);
    backrest.operation = backrest.ADD;

    planning_scene_interface.addCollisionObjects({seat, backrest});

    RCLCPP_INFO(this->get_logger(), "Added tilted chair to PlanningScene.");
  }

  void openGripper()
  {
      RCLCPP_INFO(this->get_logger(), "Opening gripper...");
      gripper_group_interface_->set_controller_name("gripper_controller");

      // 通常 Open/Close 可以用 joint 角度设置，这里假设 Open 对应 0.04 rad
      std::vector<double> open_positions = {0.04}; // 根据你夹爪的关节数量和配置调整
      gripper_group_interface_->setJointValueTarget(open_positions);

      // 执行规划和动作
      moveit::planning_interface::MoveGroupInterface::Plan plan;
      bool success = static_cast<bool>(gripper_group_interface_->plan(plan));
      if (success && !plan.trajectory_.joint_trajectory.points.empty())
      {
          auto execute_res = gripper_group_interface_->execute(plan);
          if (execute_res == moveit::core::MoveItErrorCode::SUCCESS)
          {
              RCLCPP_INFO(this->get_logger(), "Gripper opened.");
          }
          else
          {
              RCLCPP_ERROR(this->get_logger(), "Gripper execution failed.");
          }
      }
      else
      {
          RCLCPP_ERROR(this->get_logger(), "Gripper planning failed!");
      }
  }


  void planAndExecute(const geometry_msgs::msg::PoseStamped::SharedPtr &msg)
  {
    RCLCPP_INFO(this->get_logger(), "Planning and executing...");

    move_group_interface_->setStartStateToCurrentState();

    // 防御式：再次确认当前状态存在
    auto current_state = move_group_interface_->getCurrentState(0.2);
    if (!current_state)
    {
      RCLCPP_ERROR(this->get_logger(), "Current state not available before planning; abort.");
      return;
    }

    move_group_interface_->setPoseTarget(*msg);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = static_cast<bool>(move_group_interface_->plan(plan));

    if (success && !plan.trajectory_.joint_trajectory.points.empty())
    {
      auto execute_res = move_group_interface_->execute(plan);
      if (execute_res == moveit::core::MoveItErrorCode::SUCCESS)
      {
        RCLCPP_INFO(this->get_logger(), "Execution complete.");

        // 回传关节角（到你自定义的话题）
        // 获取手臂关节状态
        auto arm_joint_values = move_group_interface_->getCurrentJointValues();
        auto arm_joint_names = move_group_interface_->getJointNames();

        // 获取夹爪关节1状态
        auto gripper_joint_values = gripper_group_interface_->getCurrentJointValues();
        auto gripper_joint_names = gripper_group_interface_->getJointNames();

        // 合并
        sensor_msgs::msg::JointState js;
        js.header.stamp = this->now();

        // 先放手臂
        js.name.insert(js.name.end(), arm_joint_names.begin(), arm_joint_names.end());
        js.position.insert(js.position.end(), arm_joint_values.begin(), arm_joint_values.end());

        // 再放夹爪
        js.name.insert(js.name.end(), gripper_joint_names.begin(), gripper_joint_names.end());
        js.position.insert(js.position.end(), gripper_joint_values.begin(), gripper_joint_values.end());

        // 发布
        joint_state_pub_->publish(js);
        RCLCPP_INFO(this->get_logger(), "Joint states published to moveit_joint_states.");
      }
      else
      {
        RCLCPP_ERROR(this->get_logger(), "Execution failed with error code.");
      }
    }
    else
    {
      RCLCPP_ERROR(this->get_logger(), "Planning failed!");
    }
  }

private:
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;      // 你自定义的回传话题
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_states_seed_pub_; // 给 MoveIt 的 joint_states（只在需要时播一次）
  rclcpp::TimerBase::SharedPtr timer_;

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_;
  moveit::planning_interface::MoveGroupInterfacePtr move_group_interface_;
  moveit::planning_interface::MoveGroupInterfacePtr gripper_group_interface_;

  geometry_msgs::msg::PoseStamped last_pose_;
  bool last_pose_received_;
  bool seed_published_;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>("sim_pose_planner_base");

  // 注意：确保你的 SRDF/MoveIt 配置中有 "Arm" 与 "Gripper" 这两个规划组
  auto mgi1 = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node, "Arm");
  auto mgi2 = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node, "Gripper");
  auto planner = std::make_shared<SimPosePlanner>(mgi1, mgi2);

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(planner);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
