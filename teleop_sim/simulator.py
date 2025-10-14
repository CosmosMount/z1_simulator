from isaacgym import gymapi
from isaacgym import gymutil

import os
import time
import math
import ikpy
import numpy as np
import pinocchio as pin

from ikpy.chain import Chain
from numpy.linalg import norm,solve
from scipy.spatial.transform import Rotation as R

class z1Simulator:

    def __init__(self):

        self.gym = gymapi.acquire_gym()
        self.create_sim()
        self.create_env()
        self.create_viewer()

        self.build_ground()
        # self.build_objects()

        self.initialize_arm()

        self.moving_to_target = False
        self.target_reached = False
        self.steps_count = 0

    def create_sim(self):

        sim_params = gymapi.SimParams()
        sim_params.dt = 1 / 60
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.max_gpu_contact_pairs = 8388608
        sim_params.physx.contact_offset = 0.001
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = False

        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")
        
    def create_viewer(self):

        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise Exception("Failed to create viewer")

        cam_pos = gymapi.Vec3(1, 1, 1)
        look_at = gymapi.Vec3(-0.6, 0, 1)

        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, look_at)

        self.left_cam_offset = np.array([0, 0.033, 0])
        self.right_cam_offset = np.array([0, -0.033, 0])
        self.cam_lookat_offset = np.array([1, 0, 0])
        self.cam_pos = np.array([-2, 0, 1])

        # create left 1st preson viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280
        camera_props.height = 720
        self.left_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(self.left_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + self.left_cam_offset)),
                                     gymapi.Vec3(*(self.cam_pos + self.left_cam_offset + self.cam_lookat_offset)))
        
        # create right 1st preson viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280
        camera_props.height = 720
        self.right_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(self.right_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + self.right_cam_offset)),
                                     gymapi.Vec3(*(self.cam_pos + self.right_cam_offset + self.cam_lookat_offset)))

    def create_env(self):

        num_envs = 1
        num_per_row = int(math.sqrt(num_envs))
        env_spacing = 1.25
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        np.random.seed(0)

        self.env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

    def build_ground(self):

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)
    
    def build_objects(self):

        # Load table assets
        table_dims = gymapi.Vec3(0.4, 0.4, 0.3)  # 长、宽、高
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

        # Create the first table
        table1_pose = gymapi.Transform()
        table1_pose.p = gymapi.Vec3(0.0, 0.5, table_dims.z/2)
        self.gym.create_actor(self.env, table_asset, table1_pose, "table1", 0, 0)

        # Create the second table
        table2_pose = gymapi.Transform()
        table2_pose.p = gymapi.Vec3(0.5, 0.0, table_dims.z/2)
        self.gym.create_actor(self.env, table_asset, table2_pose, "table2", 0, 0)

        # Load block assets
        block_dims = gymapi.Vec3(0.05, 0.02, 0.1)  # 立方体
        asset_options = gymapi.AssetOptions()
        block_asset = self.gym.create_box(self.sim, block_dims.x, block_dims.y, block_dims.z, asset_options)

        # Place the block on the table
        block_pose = gymapi.Transform()
        block_pose.p = gymapi.Vec3(0.5, 0.0, table_dims.z + block_dims.z/2)
        block_actor = self.gym.create_actor(self.env, block_asset, block_pose, "block", 0, 0)

        # Set block physical properties
        props = self.gym.get_actor_rigid_body_properties(self.env, block_actor)
        props[0].mass = 0.1  # Mass
        self.gym.set_actor_rigid_body_properties(self.env, block_actor, props)

        props = self.gym.get_actor_rigid_shape_properties(self.env, block_actor)
        props[0].friction = 10.0
        self.gym.set_actor_rigid_shape_properties(self.env, block_actor, props)

        block_transform = self.gym.get_rigid_transform(self.env, block_actor)
        print(f"Block position: {block_transform.p.x - block_dims.x/2}, {block_transform.p.y - block_dims.y/2}, {block_transform.p.z - block_dims.z/2}")


        ball_density = 1.0
        ball_radius = 0.02

        # Create a sphere asset for the ball
        asset_options = gymapi.AssetOptions()
        asset_options.density = 0.5
        asset_options.fix_base_link = False  # Movable
        ball_asset = self.gym.create_sphere(self.sim, ball_radius, asset_options)

        ball_pose = gymapi.Transform()
        ball_pose.p = gymapi.Vec3(0.5, 0.0, table_dims.z + ball_radius)

        ball_actor = self.gym.create_actor(self.env, ball_asset, ball_pose, "ball", 0, 0)

        props = self.gym.get_actor_rigid_body_properties(self.env, ball_actor)
        props[0].mass = (4/3) * 3.14159 * ball_radius**3 * ball_density
        self.gym.set_actor_rigid_body_properties(self.env, ball_actor, props)
    
    def initialize_arm(self):

        asset_root = "../assets/z1/urdf"
        asset_file = "z1.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.disable_gravity = False
        asset_options.armature = 0.01
        asset_options.use_mesh_materials = True  
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX 
        asset_options.override_com = True 
        asset_options.override_inertia = True 
        asset_options.vhacd_enabled = True 
        asset_options.vhacd_params = gymapi.VhacdParams() 
        asset_options.vhacd_params.resolution = 300000 

        self.asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 0)
        pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)

        self.actor = self.gym.create_actor(self.env, self.asset, pose, "z1", 0, -1)

        self.num_dofs = self.gym.get_asset_dof_count(self.asset)
        print(f"Number of DOFs: {self.num_dofs}")
        dof_props = self.gym.get_actor_dof_properties(self.env, self.actor)
        self.lower_limits = dof_props['lower']
        self.upper_limits = dof_props['upper']

        dof_states = self.gym.get_actor_dof_states(self.env, self.actor, gymapi.STATE_ALL)
        self.dof_targets = dof_states['pos'].copy()

        urdf_path = os.path.join(asset_root, asset_file)
        self.pin_model = pin.buildModelFromUrdf(urdf_path)
        self.pin_data  = self.pin_model.createData()

        self.q_pin = pin.neutral(self.pin_model)
        self.q_home = pin.neutral(self.pin_model)

    def step(self,target,head_rmat=None,gripper_angle=0.0):

        if self.gym.query_viewer_has_closed(self.viewer):
            print("Viewer closed, exiting...")
            self.end()
            exit(0)

        self.draw_point_np(target[:3, 3])

        dof_states = self.gym.get_actor_dof_states(self.env, self.actor, gymapi.STATE_ALL)
        self.q_pin = dof_states['pos'].copy()

        q = self.inverse_kinematics(target)
        self.dof_targets[:6] = q[:6].astype(np.float32)
        if (target[2,3] < 0):
            self.dof_targets[:6] = self.q_home[:6]

        self.dof_targets[6] = gripper_angle

        self.dof_targets = np.clip(self.dof_targets, self.lower_limits, self.upper_limits)
        self.gym.set_actor_dof_position_targets(self.env, self.actor, self.dof_targets)

        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        curr_lookat_offset = self.cam_lookat_offset @ head_rmat.T
        curr_left_offset = self.left_cam_offset @ head_rmat.T
        curr_right_offset = self.right_cam_offset @ head_rmat.T

        self.gym.set_camera_location(self.left_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + curr_left_offset)),
                                     gymapi.Vec3(*(self.cam_pos + curr_left_offset + curr_lookat_offset)))
        self.gym.set_camera_location(self.right_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + curr_right_offset)),
                                     gymapi.Vec3(*(self.cam_pos + curr_right_offset + curr_lookat_offset)))
        left_image = self.gym.get_camera_image(self.sim, self.env, self.left_camera_handle, gymapi.IMAGE_COLOR)
        right_image = self.gym.get_camera_image(self.sim, self.env, self.right_camera_handle, gymapi.IMAGE_COLOR)
        left_image = left_image.reshape(left_image.shape[0], -1, 4)[..., :3]
        right_image = right_image.reshape(right_image.shape[0], -1, 4)[..., :3]

        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

        return left_image, right_image

    def end(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
    
    
    def inverse_kinematics(self, target):

        JOINT_ID = 7
        oMdes = pin.SE3(target[:3, :3], target[:3,3])

        q = self.q_pin.copy()
        eps = 1e-4
        IT_MAX = 1000
        DT = 1e-4
        damp = 1e-12

        for _ in range(IT_MAX):
            # Forward kinematics to compute the current wrist pose
            pin.forwardKinematics(self.pin_model, self.pin_data, q)
            # Calculate the current wrist pose in the world frame
            iMd = self.pin_data.oMi[JOINT_ID].actInv(oMdes)
            # Calculate the error vector in the Lie algebra space
            err = pin.log(iMd).vector

            # Check if the error is within the acceptable range
            if norm(err) < eps:
                success = True
                break

            # Calculate the Jacobian matrix for the wrist joint
            J = pin.computeJointJacobian(self.pin_model, self.pin_data, q, JOINT_ID)
            # Calculate the Jacobian matrix in the Lie algebra space
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)
            # Use the damped least squares method to compute the joint velocity
            v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
            # Update the joint positions using the computed velocity
            q = pin.integrate(self.pin_model, q, v * DT)

        self.q_pin = q
        return q
    
    def get_wrist_pose(self):
        # Get the wrist pose in the world frame
        wrist_pose = self.gym.get_rigid_transform(self.env, self.actor, 7)
        return np.array([wrist_pose.p.x, wrist_pose.p.y, wrist_pose.p.z,
                         wrist_pose.r.x, wrist_pose.r.y, wrist_pose.r.z, wrist_pose.r.w])

    def draw_point_np(self, pos_np, radius=0.05, color=(1.0, 0.3, 0.3)):
        pos_vec3 = gymapi.Vec3(*pos_np)
        color_vec3 = gymapi.Vec3(*color)
        # gymutil.draw_sphere(radius=radius,
        #                     pose=gymapi.Transform(p=pos_vec3),
        #                     color=color_vec3,
        #                     gym=self.gym,
        #                     viewer=self.viewer,
        #                     sim=self.sim)
        self.gym.clear_lines(self.viewer)
        sphere_geom = gymutil.WireframeSphereGeometry(radius=0.05, num_lats=10, num_lons=10, color=color)
        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.env, gymapi.Transform(p=pos_vec3))

    
if __name__ == "__main__":
    simulator = z1Simulator()
    try:
        while True:
            simulator.gym.simulate(simulator.sim)
            simulator.step([0.5,0,0.35,0.7071, 0.7071, 0.0, 0.0])
    except KeyboardInterrupt:
        print("Closing simulator...")
    finally:
        simulator.end()
