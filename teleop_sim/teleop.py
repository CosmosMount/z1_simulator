import numpy as np
from pytransform3d import rotations
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore

from simulator import z1Simulator
from tracker import VRTracker
from processor import VuerPreprocessor
from utils import link06_init_pose

import os
import csv


class VuerTeleop:
    def __init__(self):
        self.resolution = (720, 1280)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
        image_queue = Queue()
        toggle_streaming = Event()
        self.frame_idx = 0
        self.gripper_angle = 0.0
        self.head_rmat = np.eye(3)
        self.processor = VuerPreprocessor()
        self.simulator = z1Simulator()
        self.tracker = VRTracker(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming, ngrok=True)

    def step(self):
        if self.tracker.connected.value:
            self.processor.connected = True
            head_mat, raw_r_mat, right_controller_mat, left_controller_mat = self.processor.process(self.tracker)
            # print(f"Right Controller Matrix: {right_controller_mat}")
            # target = np.concatenate([right_controller_mat[:3,3], rotations.quaternion_from_matrix(right_controller_mat[:3, :3])[[1, 2, 3, 0]]])
            # print(f"Target Pose: {target}")
            if self.tracker.right_botton_a.value:
                self.gripper_angle -= 0.05
            elif self.tracker.right_botton_b.value:
                self.gripper_angle += 0.05
            target = right_controller_mat
            self.head_rmat = head_mat[:3, :3]
            self.frame_idx += 1
            right_pos = right_controller_mat[:3, 3]

            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([self.frame_idx] + head_mat[:3,3].tolist() + right_pos.tolist() + raw_r_mat[:3, 3].tolist())
        else:
            target = np.array([[1,0,0,0.037],
                      [0,1,0,0],
                      [0,0,1,0.17],
                      [0,0,0,1]])
        left_img, right_img = self.simulator.step(target, self.head_rmat, gripper_angle=self.gripper_angle)
        np.copyto(self.img_array, np.hstack((left_img, right_img)))

if __name__ == "__main__":
    csv_file = "controller_log.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['frame', 'head_x', 'head_y', 'head_z', 'right_x', 'right_y', 'right_z', 'raw_x', 'raw_y', 'raw_z'])
    teleop = VuerTeleop()
    try:
        while True:
            teleop.step()
    except KeyboardInterrupt:
        print("Exiting Vuer Teleop")
        teleop.simulator.end()
        exit(0)
