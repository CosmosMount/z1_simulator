import math
import numpy as np

from utils import grd_yup2grd_zup, M_bias, mat_update, fast_mat_inv

class VuerPreprocessor:
    def __init__(self):
        self.initialized = False
        self.connected = False
        self.vuer_right_controller_mat = np.eye(4)
        self.vuer_left_controller_mat = np.eye(4)
        self.T_align_right = np.eye(4)
        self.vuer_head_mat = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 1.5],
                                  [0, 0, 1, -0.2],
                                  [0, 0, 0, 1]])
    def process(self, tracker):
        self.vuer_head_mat = mat_update(self.vuer_head_mat, tracker.head_matrix.copy())
        self.vuer_right_controller_mat = mat_update(self.vuer_right_controller_mat, tracker.right_controller.copy())
        self.vuer_left_controller_mat = mat_update(self.vuer_left_controller_mat, tracker.left_controller.copy())

        head_mat = grd_yup2grd_zup @ self.vuer_head_mat @ fast_mat_inv(grd_yup2grd_zup)
        self.vuer_right_controller_mat = grd_yup2grd_zup @ self.vuer_right_controller_mat @ fast_mat_inv(grd_yup2grd_zup)
        self.vuer_left_controller_mat = grd_yup2grd_zup @ self.vuer_left_controller_mat @ fast_mat_inv(grd_yup2grd_zup)

        # if self.connected:
        #     if not self.initialized:
        #         self.T_align_right = T_base_to_link06 @ fast_mat_inv(self.vuer_right_controller_mat)
        #         self.initialized = True

        rel_right_controller_mat = fast_mat_inv(head_mat) @ self.vuer_right_controller_mat + M_bias
        rel_left_controller_mat = self.vuer_left_controller_mat
        raw_right_mat = self.vuer_right_controller_mat.copy()

        return head_mat, raw_right_mat, rel_right_controller_mat, rel_left_controller_mat
    
    