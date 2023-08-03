# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from omniisaacgymenvs.sim2real.ur10 import RealWorldUR10
from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
from omniisaacgymenvs.tasks.shared.reacher import ReacherTask
from omniisaacgymenvs.robots.articulations.views.ur10_view import UR10View
from omniisaacgymenvs.robots.articulations.ur10 import UR10
from omniisaacgymenvs.robots.articulations.views.box_view import BoxView
from omniisaacgymenvs.robots.articulations.Box import Box
from omniisaacgymenvs.robots.articulations.views.engine_view import EngineView
from omniisaacgymenvs.robots.articulations.engine import Engine
from omniisaacgymenvs.robots.articulations.views.realsense_view import RealView
from omniisaacgymenvs.robots.articulations.realsense import Real

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch import *
from omni.isaac.gym.vec_env import VecEnvBase

import random
import numpy as np
import torch
import math




class UR10ReacherTask(ReacherTask):
    def __init__(
        self,
        name: str,
        sim_config: SimConfig,
        env: VecEnvBase,
        offset=None
    ) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self.obs_type = self._task_cfg["env"]["observationType"]
        if not (self.obs_type in ["full"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [full]")
        print("Obs type:", self.obs_type)
        self.num_obs_dict = {
            "full": 32,
            # 6: UR10 joints position (action space)
            # 6: UR10 joints velocity
            # 3: goal position
            # 4: goal rotation
            # 4: goal relative rotation
            # 6: previous action
            # 7: previous goal position
            # 8: priority tensor
            # 9: Time difference tensor
            # 10: tolerance_timer_1
        }

        self.object_scale = torch.tensor([1.0] * 3)
        self.goal_scale = torch.tensor([2.0] * 3)

        self._num_observations = self.num_obs_dict[self.obs_type]
        self._num_actions = 7
        self._num_states = 0

        ## defining slow and fast targets ##

        self.slow_target = None
        self.fast_target = None
        self.current_target = None

        ### Current reset tensor
        self.current_reset_num = None

        ##########


        pi = math.pi
        if self._task_cfg['safety']['enabled']:
            # Depends on your real robot setup
            self._dof_limits = torch.tensor([[
                [np.deg2rad(-135), np.deg2rad(135)],
                [np.deg2rad(-180), np.deg2rad(-60)],
                [np.deg2rad(0), np.deg2rad(180)],
                [np.deg2rad(-180), np.deg2rad(0)],
                [np.deg2rad(-180), np.deg2rad(0)],
                [np.deg2rad(-180), np.deg2rad(180)],
            ]], dtype=torch.float32, device=self._cfg["sim_device"])
        else:
            # For actions
            self._dof_limits = torch.tensor([[
                [-2*pi, 2*pi],           # [-2*pi, 2*pi],
                [-pi + pi/8, 0 - pi/8],  # [-2*pi, 2*pi],
                [-pi + pi/8, pi - pi/8], # [-2*pi, 2*pi],
                [-pi, 0],                # [-2*pi, 2*pi],
                [-pi, pi],               # [-2*pi, 2*pi],
                [-2*pi, 2*pi],           # [-2*pi, 2*pi],
            ]], dtype=torch.float32, device=self._cfg["sim_device"])
            # The last action space cannot be [0, 0]
            # It will introduce the following error:
            # ValueError: Expected parameter loc (Tensor of shape (2048, 6)) of distribution Normal(loc: torch.Size([2048, 6]), scale: torch.Size([2048, 6])) to satisfy the constraint Real(), but found invalid values

        ReacherTask.__init__(self, name=name, env=env)



        # Setup Sim2Real
        sim2real_config = self._task_cfg['sim2real']
        if sim2real_config['enabled'] and self.test and self.num_envs == 1:
            self.act_moving_average /= 5 # Reduce moving speed
            self.real_world_ur10 = RealWorldUR10(
                sim2real_config['fail_quietely'],
                sim2real_config['verbose']
            )
        return






    def get_num_dof(self):
        return self._arms.num_dof

    def get_arm(self):
        ur10 = UR10(prim_path=self.default_zero_env_path + "/ur10", name="UR10")
        self._sim_config.apply_articulation_settings(
            "ur10",
            get_prim_at_path(ur10.prim_path),
            self._sim_config.parse_actor_config("ur10"),
        )

    def get_arm_view(self, scene):
        arm_view = UR10View(prim_paths_expr="/World/envs/.*/ur10", name="ur10_view")
        scene.add(arm_view._end_effectors)
        return arm_view

    def get_box(self):
        box = Box(prim_path=self.default_zero_env_path + "/Box", name="BOX")
        self._sim_config.apply_articulation_settings(
            "Box",
            get_prim_at_path(box.prim_path),
            self._sim_config.parse_actor_config("Box"),
        )

    def get_box_view(self, scene):
        box_view1 = BoxView(prim_paths_expr="/World/envs/.*/Box", name="box_view")
        #scene.add(box_view1)
        return box_view1
    
    def get_engine(self):
        box = Engine(prim_path=self.default_zero_env_path + "/Engine", name="ENGINE")
        self._sim_config.apply_articulation_settings(
            "Engine",
            get_prim_at_path(box.prim_path),
            self._sim_config.parse_actor_config("Engine"),
        )

    def get_engnie_view(self, scene):
        engine_view1 = EngineView(prim_paths_expr="/World/envs/.*/Engine", name="engine_view")
        #scene.add(box_view1)
        return engine_view1
    
    def get_realsense(self):
        camera = Real(prim_path=self.default_zero_env_path + "/Real", name="REAL")
        self._sim_config.apply_articulation_settings(
            "Real",
            get_prim_at_path(camera.prim_path),
            self._sim_config.parse_actor_config("Real"),
        )

    def get_realsense_view(self, scene):
        real_view1 = RealView(prim_paths_expr="/World/envs/.*/Real", name="realsense_view")
        #scene.add(box_view1)
        return real_view1


    def get_object_displacement_tensor(self):
        return torch.tensor([0.0, 0.003, 0.0], device=self.device).repeat((self.num_envs, 1)) # Change from 0.05 to 0.003 for Picam instaead of the cube


    def get_observations(self):
        self.arm_dof_pos = self._arms.get_joint_positions()
        self.arm_dof_vel = self._arms.get_joint_velocities()

        if self.obs_type == "full_no_vel":
            self.compute_full_observations(True)
        elif self.obs_type == "full":
            self.compute_full_observations()
        else:
            print("Unkown observations type!")

        ### Verbose ###

        # verb_config = self._task_cfg['verbose']
        # assert verb_config
        # if verb_config['enabled']:
        #     print("testing!!!!!!!")
        #     print("action moving average = %s ", self.act_moving_average)

        #########

        observations = {
            self._arms.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    # def get_extras(self):

    #     return {"extras": self.extras}


    def get_reset_target_new_pos(self, n_reset_envs, priority_tensor, reset_envs, all_resets, all_prev_resets):
        # Randomly generate goal positions, although the resulting goal may still not be reachable.
        new_pos = torch_rand_float(-1, 1, (n_reset_envs, 3), device=self.device)

        def newpos():
            return torch_rand_float(-0.7, 0.7 , (1, 3), device=self.device)


        target_poses = [[0.143423, -0.13423, 0.343423],[0.943423, -0.13423, 0.343423], [0.943423, -0.93423, 0.343423],[0.143423, -0.93423, 0.343423]] #,[-0.143423, -0.13423, 0.343423], [-0.443423, -0.13423, 0.343423]

        # Slow Target, Fast Target

        self.slow_target = torch.tensor([0.443423, -0.13423, 0.343423], device=self.device)
        self.fast_target = torch.tensor([-0.543423, -0.13423, 0.343423], device=self.device)

        # 6 targets

        self.first_target = torch.tensor([0.143423, -0.13423, 0.343423], device=self.device)
        self.second_target = torch.tensor([0.943423, -0.13423, 0.343423], device=self.device)
        self.third_target = torch.tensor([0.943423, -0.93423, 0.343423], device=self.device)
        self.fourth_target = torch.tensor([0.143423, -0.93423, 0.343423], device=self.device)
        self.fifth_target = torch.tensor([-0.43423, -0.93423, 0.343423], device=self.device)
        self.sixth_target = torch.tensor([-0.943423, -0.93423, 0.343423], device=self.device)
        self.seventh_target = torch.tensor([0.543423, 0.43423, 0.343423], device=self.device)
        self.eighth_target = torch.tensor([0.543423, 0.73423, 0.343423], device=self.device)
        self.nineth_target = torch.tensor([0.743423, 0.73423, 0.343423], device=self.device)
        self.tenth_target = torch.tensor([0.943423, 0.73423, 0.343423], device=self.device)
        self.eleventh_target = torch.tensor([0.743423, 0.73423, 0.443423], device=self.device)
        self.twelth_target = torch.tensor([0.743423, 0.23423, 0.443423], device=self.device)
        






        # Making validation points


        six_points = [self.fifth_target, self.sixth_target ,self.first_target, self.second_target, self.third_target, self.fourth_target]
        temp = [newpos() for _ in range(6)]
        

        if self.test:
            if not self.validation:
                # self.validation = [six_points for x in range(self.num_envs)]
                self.validation = []
                loaded_valid_list = np.load('/RLrepo/OmniIsaacGymEnvs-UR10Reacher/omniisaacgymenvs/validation/tensors.npy', allow_pickle=True)
                for item in loaded_valid_list:
                    self.validation.append([torch.from_numpy(numpy_array) for numpy_array in item])
                # self.validation = [[newpos() for _ in range(6)] for x in range(200)]
                self.current_target_points = self.validation[0]
                # print(self.validation)


        if self.test:

            if not torch.equal(all_resets, all_prev_resets):
                # print(f"test-----------{self.current_target_points}")
                for num, pts in enumerate(self.validation):
                    if torch.equal(torch.stack(pts), torch.stack(self.current_target_points)) and num != 99:
                        # print(num, self.validation[num])
                        self.current_target_points = self.validation[num+1]
                        break
                    elif num == 99:
                        print("Last validation point reached and exiting")
                        self.current_target_points = None
                
            # print(f"-----------{self.current_target_points}")
            # for num,items in enumerate(zip(all_resets, all_prev_resets)):
            #     if items[0].item() != items[1].item():
            #         self.validation[num] = temp



        new_pos_2 = torch.zeros((n_reset_envs, 3), device=self.device)

        # target_points = [ self.fifth_target, self.sixth_target ,self.first_target, self.second_target, self.third_target, self.fourth_target]
        target_points = [ self.seventh_target, self.eighth_target, self.nineth_target, self.tenth_target, self.eleventh_target, self.twelth_target]
        priority_list = []
        target_env_pts = []


        # sorting priority tensor with reset tensor
        for env_id_tensor in reset_envs:
            env_int_id = env_id_tensor.item()
            priority_list.append(priority_tensor[env_int_id])
            #target_env_pts.append(self.validation[env_int_id])


        priority_n_reset = torch.stack((priority_list), dim=0)



        for row_num,row in enumerate(priority_n_reset):
            for column_num,column in enumerate(row):
                if column==True:
                    # new_pos_2[row_num] = target_env_pts[row_num][column_num]
                    new_pos_2[row_num] = target_points[column_num]
                    # new_pos_2[row_num] = self.current_target_points[column_num]


        # Change the prority_tensor for the next reset

        new_priority_tensor = priority_tensor.clone()

        true_tensor = torch.tensor([True], device = self.device)
        false_tensor = torch.tensor([False], device = self.device)


        for env_idtensor in reset_envs:
            env_int = env_idtensor.item()
            for num, env_bool in enumerate(new_priority_tensor[env_int]):
                #print(env_bool, new_priority_tensor[env_int, num])
                if env_bool == true_tensor and  new_priority_tensor[env_int,num] != new_priority_tensor[env_int, -1]:
                    new_priority_tensor[env_int, num] = false_tensor
                    new_priority_tensor[env_int, num+1] = true_tensor
                    break
                elif env_bool == true_tensor and  new_priority_tensor[env_int,num] == new_priority_tensor[env_int,-1]:
                    new_priority_tensor[env_int, num] = false_tensor
                    new_priority_tensor[env_int, 0] = true_tensor
                    break



       # Current target for speed changes

        def find_current_pos(item):

            if item == self.first_target.tolist():
                current_target = 1
            elif item == self.second_target.tolist():
                current_target = 2
            elif item == self.third_target.tolist():
                current_target = 3
            elif item == self.fourth_target.tolist():
                current_target = 4
            elif item == self.fifth_target.tolist():
                current_target = 5
            else:
                current_target = 6
            return current_target


        target_pose = torch.tensor(random.choice(target_poses), device=self.device)

        current_pos = torch.zeros(n_reset_envs, device=self.device)

        for num,x in enumerate(new_pos_2):
            pos = find_current_pos(x.tolist())
            current_pos[num] = pos




        #new_pos = torch.full((n_reset_envs, 3),random.choice(target_poses) , device=self.device)
        #new_pos = target_pose.repeat(n_reset_envs,1)


        #cur_mit_env = torch.tensor([self.current_target for x in range(n_reset_envs)], device=self.device)

        if self._task_cfg['sim2real']['enabled'] and self.test and self.num_envs == 1:
            # Depends on your real robot setup
            new_pos[:, 0] = torch.abs(new_pos[:, 0] * 0.1) + 0.35
            new_pos[:, 1] = torch.abs(new_pos[:, 1] * 0.1) + 0.35
            new_pos[:, 2] = torch.abs(new_pos[:, 2] * 0.5) + 0.3
        else:
            new_pos[:, 0] = new_pos[:, 0] * 0.4 + 0.5 * torch.sign(new_pos[:, 0])
            new_pos[:, 1] = new_pos[:, 1] * 0.4 + 0.5 * torch.sign(new_pos[:, 1])
            new_pos[:, 2] = torch.abs(new_pos[:, 2] * 0.8) + 0.1
        if self._task_cfg['safety']['enabled']:
            new_pos[:, 0] = torch.abs(new_pos[:, 0]) / 1.25
            new_pos[:, 1] = torch.abs(new_pos[:, 1]) / 1.25

        if self._task_cfg['sim2real']['enabled'] and self.test and self.num_envs == 1:
            # Depends on your real robot setup
            new_pos_2[:, 0] = torch.abs(new_pos_2[:, 0] * 0.1) + 0.35
            new_pos_2[:, 1] = torch.abs(new_pos_2[:, 1] * 0.1) + 0.35
            new_pos_2[:, 2] = torch.abs(new_pos_2[:, 2] * 0.5) + 0.3
        else:
            new_pos_2[:, 0] = new_pos_2[:, 0] * 0.4 + 0.5 * torch.sign(new_pos_2[:, 0])
            new_pos_2[:, 1] = new_pos_2[:, 1] * 0.4 + 0.5 * torch.sign(new_pos_2[:, 1])
            new_pos_2[:, 2] = torch.abs(new_pos_2[:, 2] * 0.8) + 0.1
        if self._task_cfg['safety']['enabled']:
            new_pos_2[:, 0] = torch.abs(new_pos_2[:, 0]) / 1.25
            new_pos_2[:, 1] = torch.abs(new_pos_2[:, 1]) / 1.25
        # print(new_pos_2)
        # new_pos_2 : with priorities on defenite points, new_pos : random points
        return new_pos,  current_pos, target_points, new_pos_2 , new_priority_tensor

    def compute_full_observations(self, no_vel=False):
        if no_vel:
            raise NotImplementedError()
        else:
            # There are many redundant information for the simple Reacher task, but we'll keep them for now.
            self.obs_buf[:, 0:self.num_arm_dofs] = unscale(self.arm_dof_pos[:, :self.num_arm_dofs],
                self.arm_dof_lower_limits, self.arm_dof_upper_limits)
            self.obs_buf[:, self.num_arm_dofs:2*self.num_arm_dofs] = self.vel_obs_scale * self.arm_dof_vel[:, :self.num_arm_dofs]
            base = 2 * self.num_arm_dofs
            self.obs_buf[:, base+0:base+3] = self.goal_pos
            self.obs_buf[:, base+3:base+7] = self.goal_rot
            self.obs_buf[:, base+7:base+11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
            self.obs_buf[:, base+11:base+18] = self.actions
            self.obs_buf[:, base+18:base+19] = self.tolerance_timer_1.unsqueeze(1)
            self.obs_buf[:, base+19:base+20] = self.act_moving_average
            # self.obs_buf[:, base+18:base+19] = self.cur_goal_pos.unsqueeze(1)
            # self.obs_buf[:, base+19:base+25] = self.priority
            #self.obs_buf[:, base+18:base+19] = self.time_diff


    def send_joint_pos(self, joint_pos):
        self.real_world_ur10.send_joint_pos(joint_pos)
