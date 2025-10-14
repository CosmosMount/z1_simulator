import os
import rospy
import numpy as np
import pinocchio as pin
from std_msgs.msg import Float32MultiArray, Int32MultiArray

# controller implemented based on b2z1

Xoffset = -4.4
Yoffset = 1.2
 
class z1Controller:
    def __init__(self):
        # load pinocchio model
        asset_root = "../assets/z1/urdf"
        asset_file = "z1.urdf"
        urdf_path = os.path.join(asset_root, asset_file)
        self.z1_dyn_model = pin.buildModelFromUrdf(urdf_path) 
        self.z1_dyn_data = self.z1_dyn_model.createData()
        self.ee_frame_id = self.z1_dyn_model.getFrameId("gripperStator", pin.FrameType.BODY)
        self.base_link_id = self.z1_dyn_model.getFrameId("link00", pin.FrameType.BODY)

        # initialize joints
        self.joint_cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_positions =  [ -0.0,  0.8,  -1.5,  0.0, 0.8, -1.5, 
                                  -0.0,  0.8,  -1.5,  0.0, 0.8, -1.5,
                                  0.0000,  1.4800, -0.6300, -0.8400,  0.0000,  1.5700, 0]

        rospy.init_node('teleop_node', anonymous=True)

        # subscribe b2 and z1 observation
        rospy.Subscriber("/b2z1_cur_joint", Float32MultiArray, self.b2z1_obs_callback) # from b2 highlevel controller, b2z1 joint state
        rospy.Subscriber("/env_obs", Float32MultiArray, self.env_obs_callback) # from motion capture, robot global pose, object global pose
        # publish z1 pos
        self.z1_target_pos_pub = rospy.Publisher('arm_target_pos', Float32MultiArray, queue_size=2)
        rospy.Timer(rospy.Duration(0.1), self.publish_target_pos)

    def b2z1_obs_callback(self, msg):
        """
        Subscriber 回调函数，用于接收 b2z1 观察数据
        """
        data = msg.data # 19 dims
        if data is not None:
            self.joint_positions = list(data) # order is matter

    def env_obs_callback(self, msg):
        """
        Subscriber 回调函数，用于接收机器人和物体的观察数据
        """
        data = msg.data
        robot_obs = data[:8]   # 8
        object_obs = data[8:]  # 4*4 = 16

        # update current poses
        if robot_obs is not None: 
            self.robot_pose[0] = float(robot_obs[0]) + Xoffset
            self.robot_pose[1] = float(robot_obs[1]) + Yoffset
            self.robot_pose[2] = float(robot_obs[2])
            self.robot_pose[3] = float(robot_obs[3])  # yaw

            # Calculate arm_base_pose: robot pose offset by 0.2m along robot's local x-axis
            robot_yaw = self.robot_pose[3]
            arm_offset_x = 0.2 * np.cos(robot_yaw)
            arm_offset_y = 0.2 * np.sin(robot_yaw)
            
            self.arm_base_pose[0] = self.robot_pose[0] + arm_offset_x
            self.arm_base_pose[1] = self.robot_pose[1] + arm_offset_y
            self.arm_base_pose[2] = self.robot_pose[2] 
            self.arm_base_pose[3:7] = list(robot_obs[4:8])  # [qx, qy, qz, qw]
        
    def publish_target_pos(self, event=None):
        msg = Float32MultiArray(data=self.joint_cmd)
        self.z1_target_pos_pub.publish(msg)

    def step(self,target,head_rmat=None,gripper_angle=0.0):
        qnow = np.array(self.joint_positions.copy()[12:19], dtype=np.float64)
        q = self.inverse_kinematics(qnow,target)
        q[6] = gripper_angle
        
    def inverse_kinematics(self, qnow, target):

        oMdes = pin.SE3(target[:3, :3], target[:3,3])
        q = qnow.copy()
  
        eps = 1e-3
        IT_MAX = 200
        damp = 1e-12
        DT = 3e-3    # 最小步长，保证平滑

        for _ in range(IT_MAX):
            pin.forwardKinematics(self.z1_dyn_model, self.z1_dyn_data, q)
            pin.updateFramePlacements(self.z1_dyn_model, self.z1_dyn_data)
            iMd = self.z1_dyn_data.oMf[self.ee_frame_id].actInv(oMdes)

            err = pin.log(iMd).vector
            self.err_norm = np.linalg.norm(err)
            
            if np.linalg.norm(err) < eps:
                break

            J = pin.computeFrameJacobian(self.z1_dyn_model, self.z1_dyn_data, q, self.ee_frame_id, pin.ReferenceFrame.LOCAL)
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)
            v = -J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pin.integrate(self.z1_dyn_model, q, v * DT)

        return q