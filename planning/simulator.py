from isaacgym import gymapi
from isaacgym import gymutil

import os
import time
import math
import json
import socket
import threading
import numpy as np

import pinocchio as pin
from numpy.linalg import norm,solve
from scipy.spatial.transform import Rotation as R


class z1_simulator:

    def __init__(self, host='127.0.0.1', port=9999):
        self.gym = gymapi.acquire_gym()
        self.create_sim()
        self.create_env()
        self.create_viewer()

        self.build_ground()
        # self.build_chair()

        self.initialize_arm()
        self.initialize_events()

        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect((host, port))
        self.running = True
        self.listener_thread = threading.Thread(target=self.listen_cmd, daemon=True)
        self.listener_thread.start()

        self.received_dof_targets = None
    
    def send_sim_state(self, state):
        try:
            msg = json.dumps({"type": "state", "data": state}) + '\n'
            self.conn.sendall(msg.encode('utf-8'))
            # print("[IsaacSim] Sent message:", msg)
        except Exception as e:
            print("[IsaacSim] Failed to send message:", e)

    def listen_cmd(self):
        buffer = ""
        while self.running:
            try:
                data = self.conn.recv(1024).decode('utf-8')
                if not data:
                    continue
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if not line.strip():
                        continue
                    msg = json.loads(line)
                    if msg["type"] in ["cmd", "joint_state"]:
                        joint_targets = msg["data"].get("position") or msg["data"].get("joint_targets")
                        if joint_targets and len(joint_targets) == self.num_dofs:
                            self.received_dof_targets = np.array(joint_targets)
                        else:
                            print("[IsaacSim] Received joint_targets length mismatch")
            except Exception as e:
                print("[IsaacSim] Error receiving command:", e)
                time.sleep(0.01)  # 避免线程空转占 CPU

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

        cam_pos = gymapi.Vec3(1.0, 1.0, 1.0)
        look_at = gymapi.Vec3(0.0, 0.0, 0.0)

        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, look_at)

        # # Set viewer window always on top
        # window_id = os.popen("wmctrl -l | grep 'Isaac Gym' | awk '{print $1}'").read().strip()
        # if window_id:
        #     os.system(f"wmctrl -i -r {window_id} -b add,above")

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
    
    def build_chair(self):
        # 椅座参数
        seat_size = gymapi.Vec3(0.5 / 2, 0.5 / 2, 0.05 / 2)  # 半尺寸（IsaacGym create_box 接收半尺寸）
        seat_pose = gymapi.Transform()
        seat_pose.p = gymapi.Vec3(0.5, 0.0, 0.4)  # 椅座中心位置
        seat_pose.r = gymapi.Quat()  # 无旋转

        seat_asset_options = gymapi.AssetOptions()
        seat_asset_options.fix_base_link = True  # 椅子固定，不动
        seat_asset = self.gym.create_box(self.sim, seat_size.x, seat_size.y, seat_size.z, seat_asset_options)

        self.gym.create_actor(self.env, seat_asset, seat_pose, "chair_seat", 0, 0)

        # 椅背参数
        back_size = gymapi.Vec3(0.05 / 2, 0.5 / 2, 0.6 / 2)  # 半尺寸
        back_pose = gymapi.Transform()
        back_pose.p = gymapi.Vec3(0.5 - 0.025, 0.0, 0.45)  # 椅背中心位置

        # 计算椅背15度绕Y轴旋转四元数
        angle = 15.0 * math.pi / 180.0
        q_back = gymapi.Quat.from_euler_zyx(0, -angle, 0)
        back_pose.r = q_back

        back_asset_options = gymapi.AssetOptions()
        back_asset_options.fix_base_link = True
        back_asset = self.gym.create_box(self.sim, back_size.x, back_size.y, back_size.z, back_asset_options)

        self.gym.create_actor(self.env, back_asset, back_pose, "chair_backrest", 0, 0)
    
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

        # Initialize Pinocchio model and pose
        self.q_pin = pin.neutral(self.pin_model)
        self.q_home = pin.neutral(self.pin_model)

        
    def initialize_events(self):

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_I, "input_coords")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "show_coords")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset_all")
        
        for i in range(1, 8):
            self.gym.subscribe_viewer_keyboard_event(self.viewer, getattr(gymapi, f"KEY_{i}"), f"joint_{i}")

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT_SHIFT, "shift")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_RIGHT_SHIFT, "shift")

        self.key_states = {
            "joint_1": False, "joint_2": False, "joint_3": False,
            "joint_4": False, "joint_5": False, "joint_6": False,
            "joint_7": False, "shift": False
        }
    
    def step(self):

        if self.gym.query_viewer_has_closed(self.viewer):
            print("Viewer closed, exiting...")
            self.end()
            exit(0)
        
        
        events = self.gym.query_viewer_action_events(self.viewer)
        for event in events:
            
            if event.action == "show_coords" and event.value > 0:
                # transform = self.gym.get_rigid_transform(self.env, 7)
                # current_position = np.array([transform.p.x, transform.p.y, transform.p.z], dtype=np.float32)
                # print(f"Current position: {current_position}")
                # print(transform)
                

                # Forward kinematics
                
                print(f"Current pose:{self.pin_data.oMf[7]}")
                # print(f"Current DOF states: {dof_state}")

            if event.action == "reset_all":
                self.dof_targets[:6] = self.q_home[:6]

        for event in events:
            if event.action in self.key_states:
                self.key_states[event.action] = event.value > 0

        for i in range(7):
            action_key = f"joint_{i+1}"
            if self.key_states[action_key]:
                direction = -1 if self.key_states["shift"] else 1
                if i == 0:
                    self.dof_targets[i] += direction * 0.05
                else:
                    self.dof_targets[i] += direction * 0.005
            
        if self.received_dof_targets is not None:
            # 如果接收到新的关节目标角度，更新目标
            self.dof_targets = np.array(self.received_dof_targets.copy(), dtype=np.float32)
            self.received_dof_targets = None
        self.gym.set_actor_dof_position_targets(self.env, self.actor, self.dof_targets)
        transform = self.gym.get_rigid_transform(self.env, 8)
        state = {
            "position": [transform.p.x, transform.p.y, transform.p.z],
            "rotation": [transform.r.x, transform.r.y, transform.r.z, transform.r.w]
        }
        
        self.gym.simulate(self.sim)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

        self.send_sim_state(state)

    def end(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
    

if __name__ == '__main__':
    simulator = z1_simulator()
    try:
        while True:
            simulator.step()
    except KeyboardInterrupt:
        simulator.end()
        exit(0)


