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


from abc import abstractmethod

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.prims import RigidPrimView, XFormPrim
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from omni.isaac.core.utils.torch import *
# `scale` maps [-1, 1] to [L, U]; `unscale` maps [L, U] to [-1, 1]
from omni.isaac.core.utils.torch import scale, unscale
from omni.isaac.gym.vec_env import VecEnvBase
from omni.isaac.core.utils import nucleus
from omni.isaac.core.utils.viewports import set_camera_view

import numpy as np
import torch

##############
#added from "William Rodmann"

import omni.kit.commands
from pxr import UsdGeom, Gf
from omni.isaac.isaac_sensor import _isaac_sensor
from copy import copy
from enum import Enum

class UR10_events(Enum):
    START = 0
    GOAL_REACHED = 1
    ATTACHED = 2
    DETACHED = 3
    TIMEOUT = 4
    STOP = 5
    
    NONE = 6

class UR10_states(Enum):
    STANDBY=0
    PICKING=1
    ATTACH=2
    PLACING=3
    DETACH=4
#############

BACKGROUND_STAGE_PATH = "/background"
BACKGROUND_USD_PATH = "/Isaac/Environments/Simple_Room/simple_room.usd"
assets_root_path = nucleus.get_assets_root_path()

class ReacherTask(RLTask):
    def __init__(
        self,
        name: str,
        env: VecEnvBase,
        offset=None
    ) -> None:
        """[summary]
        """
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self._task_cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self._task_cfg["env"]["reachGoalBonus"]
        self.rot_eps = self._task_cfg["env"]["rotEps"]
        self.vel_obs_scale = self._task_cfg["env"]["velObsScale"]

        self.reset_position_noise = self._task_cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self._task_cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self._task_cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self._task_cfg["env"]["resetDofVelRandomInterval"]

        self.arm_dof_speed_scale = self._task_cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self._task_cfg["env"]["useRelativeControl"]
        self.act_moving_average = self._task_cfg["env"]["actionsMovingAverage"]

        self.max_episode_length = self._task_cfg["env"]["episodeLength"]
        self.reset_time = self._task_cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self._task_cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self._task_cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self._task_cfg["env"].get("averFactor", 0.1)

        self.dt = 1.0 / 60
        control_freq_inv = self._task_cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        RLTask.__init__(self, name, env)

        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # Indicates which environments should be reset
        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = torch.tensor(self.av_factor, dtype=torch.float, device=self.device)
        self.total_successes = 0
        self.total_resets = 0
        
        ####
        #added from "William Rodmann"

        self.paltform_pos=torch.tensor([0.5,0.6,0.3],  device=self.device)

        # #Short Gripper

        # self._is_moving = False

        # self.sm= {}
        # for s in UR10_states:
        #     self.sm[s] = {}
        #     for e in UR10_events:
        #         self.sm[s][e] = self._empty
        #         self.thresh[s] = 0

        # self.sm[UR10_states.STANDBY][UR10_events.GOAL_REACHED] = self._standby_start()
        # self.sm[UR10_states.STANDBY][UR10_events.GOAL_REACHED] = self._standby_goal_reached
        # self.thresh[UR10_states.STANDBY] = 3

        # self.sm[UR10_states.PICKING][UR10_events.GOAL_REACHED] = self._picking_goal_reached
        # self.sm[UR10_states.PICKING][UR10_events.NONE] = self._picking_no_event
        # self.thresh[UR10_states.PICKING] = 1

        # #self.sm[UR10_states.FLIPPING][UR10_events.GOAL_REACHED] = self._flipping_goal_reached
        # #self.thresh[UR10_states.FLIPPING] = 2

        # self.sm[UR10_states.PLACING][UR10_events.GOAL_REACHED] = self._placing_goal_reached
        # self.thresh[UR10_states.PLACING] = 0

        # self.sm[UR10_states.ATTACH][UR10_events.GOAL_REACHED] = self._attach_goal_reached
        # self.sm[UR10_states.ATTACH][UR10_events.ATTACHED] = self._attach_attached
        # self.thresh[UR10_states.ATTACH] = 0

        # self.sm[UR10_states.DETACH][UR10_events.GOAL_REACHED] = self._detach_goal_reached
        # self.sm[UR10_states.DETACH][UR10_events.DETACHED] = self._detach_detached
        # self.thresh[UR10_states.DETACH] = 0

        # self.current_state = UR10_states.STANDBY
        # self.previous_state = -1
        # self._physx_query_interface = omni.physx.get_physx_scene_query_interface()

        ######
        return

    def set_up_scene(self, scene: Scene) -> None:
        self._stage = get_current_stage()
        self._assets_root_path = '/home/willi/Dokumente/Omniverse-Pick-and-Place/OmniIsaacGymEnvs-UR10Reacher/Isaac/2022.1'
        self._ur10 =scene.add(self.get_arm())

        self.get_object()
        self.get_goal()
        self.get_platform()

        super().set_up_scene(scene)

        self._arms = self.get_arm_view(scene)
        
        scene.add(self._arms)
        ##################
        #function is used to set the location of the gripper in relation to the UR10 robot.
        self._ur10._gripper.set_translate(value=0.162) 
        #function is used to set the direction of the gripper.
        self._ur10._gripper.set_direction(value="x")
        #functions are used to set the force and torque limits of the gripper.
        self._ur10._gripper.set_force_limit(value=8.0e1)
        self._ur10._gripper.set_torque_limit(value=5.0e0)
        ##################
        self._objects = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/object/object",
            name="object_view",
            reset_xform_properties=False,
        )
        scene.add(self._objects)

        self._goals = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/goal/object",
            name="goal_view",
            reset_xform_properties=False,
        )
        scene.add(self._goals)

        # set default camera viewport position and target
        #self.set_initial_camera_params()

    # def set_initial_camera_params(self, camera_position=[3, 3, 2], camera_target=[0, 0, 0]):
    #     set_camera_view(eye=camera_position, target=camera_target, camera_prim_path="/OmniverseKit_Persp")


    @abstractmethod
    def get_num_dof(self):
        pass

    @abstractmethod
    def get_arm(self):
        pass

    @abstractmethod
    def get_arm_view(self):
        pass

    @abstractmethod
    def get_observations(self):
        pass

    @abstractmethod
    def get_reset_target_new_pos(self, n_reset_envs):
        pass

    @abstractmethod
    def send_joint_pos(self, joint_pos):
        pass

    def get_object(self):
        self.object_start_translation = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        self.object_start_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self.object_usd_path = f"{self._assets_root_path}/Isaac/Props/Blocks/block_instanceable.usd"
        add_reference_to_stage(self.object_usd_path, self.default_zero_env_path + "/object")
        obj = XFormPrim(
            prim_path=self.default_zero_env_path + "/object/object",
            name="object",
            translation=self.object_start_translation,
            orientation=self.object_start_orientation,
            scale=self.object_scale
        )
        self._sim_config.apply_articulation_settings("object", get_prim_at_path(obj.prim_path), self._sim_config.parse_actor_config("object"))

    def get_goal(self):
        self.goal_displacement_tensor = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        self.goal_start_translation = torch.tensor([0.0, 0.0, 0.0], device=self.device) + self.goal_displacement_tensor
        self.goal_start_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        self.goal_usd_path = f"{self._assets_root_path}/Isaac/Props/Blocks/block_instanceable.usd"
        add_reference_to_stage(self.goal_usd_path, self.default_zero_env_path + "/goal")
        goal = XFormPrim(
            prim_path=self.default_zero_env_path + "/goal/object",
            name="goal",
            translation=self.goal_start_translation,
            orientation=self.goal_start_orientation,
            scale=self.goal_scale
        )
        self._sim_config.apply_articulation_settings("goal", get_prim_at_path(goal.prim_path), self._sim_config.parse_actor_config("goal_object"))
        
    def get_platform(self):
        self.platform_position = self.paltform_pos
        self.platform_scale = torch.tensor([(0.3, 0.4, 0.03)], device=self.device)
        self.object_usd_path = f"{self._assets_root_path}/Isaac/Props/Blocks/platform_instanceable_help.usd"
        add_reference_to_stage(self.object_usd_path, self.default_zero_env_path + "/platform")
        platform = XFormPrim(
            prim_path=self.default_zero_env_path + "/platform/platform/Cube",
            name="platform",
            translation=self.platform_position,
           scale=self.platform_scale,
            visible=True,

        )
        self._sim_config.apply_articulation_settings("platform", get_prim_at_path(platform.prim_path), self._sim_config.parse_actor_config("platform_object"))

    def get_cube(self):
        self.cube_displacement_tensor = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        self.cube_start_translation = torch.tensor([1.0, 2.0, 3.0], device=self.device) + self.cube_displacement_tensor
        self.cube_start_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        self.cube_usd_path = f"{self._assets_root_path}/Isaac/Props/Blocks/pp_instanceable.usd"
        add_reference_to_stage(self.cube_usd_path, self.default_zero_env_path + "/cube")
        cube = XFormPrim(
            prim_path=self.default_zero_env_path + "/pp/Cube",
            name="cube",
            translation=self.cube_start_translation,
            orientation=self.cube_start_orientation,
            scale=self.cube_scale
        )
        self._sim_config.apply_articulation_settings("cube", get_prim_at_path(cube.prim_path), self._sim_config.parse_actor_config("cube_object"))

    def post_reset(self):
        self.num_arm_dofs = self.get_num_dof()
        self.actuated_dof_indices = torch.arange(self.num_arm_dofs, dtype=torch.long, device=self.device)

        self.arm_dof_targets = torch.zeros((self.num_envs, self._arms.num_dof), dtype=torch.float, device=self.device)

        self.prev_targets = torch.zeros((self.num_envs, self.num_arm_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_arm_dofs), dtype=torch.float, device=self.device)

        dof_limits = self._dof_limits
        self.arm_dof_lower_limits, self.arm_dof_upper_limits = torch.t(dof_limits[0].to(self.device))

        self.arm_dof_default_pos = torch.zeros(self.num_arm_dofs, dtype=torch.float, device=self.device)
        self.arm_dof_default_vel = torch.zeros(self.num_arm_dofs, dtype=torch.float, device=self.device)

        self.end_effectors_init_pos, self.end_effectors_init_rot = self._arms._end_effectors.get_world_poses()

        self.object_pos, self.object_rot = self._objects.get_world_poses()
        self.object_pos -= self._env_pos

        self.goal_pos, self.goal_rot = self._goals.get_world_poses()
        self.goal_pos -= self._env_pos

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self):
        end_effector_pos, end_effector_rot = self._arms._end_effectors.get_world_poses()
        self.fall_dist = 0
        self.fall_penalty = 0
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_arm_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot, self.goal_pos, self.goal_rot,
            end_effector_pos, # end_effector_rot, # TODO: use end effector rotation
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor,
        )
        #print(self.rew_buf) # print reward buffer
        self.extras['consecutive_successes'] = self.consecutive_successes.mean()

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()
            # The direct average shows the overall result more quickly, but slightly undershoots long term policy performance.
            print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes/self.total_resets))

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        self._ur10._gripper.update()

        end_effectors_pos, end_effectors_rot = self._arms._end_effectors.get_world_poses()
        object_dist = torch.norm(end_effectors_pos - self.object_pos, p=2, dim=-1)
        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids)
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)
        # Reacher tasks don't require gripper actions, disable it.
        #self.actions[:, 5] = 0.0

        if self.use_relative_control:
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.arm_dof_speed_scale * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                self.arm_dof_lower_limits[self.actuated_dof_indices], self.arm_dof_upper_limits[self.actuated_dof_indices])
        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions,
                self.arm_dof_lower_limits[self.actuated_dof_indices], self.arm_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:, self.actuated_dof_indices] + \
                (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
                self.arm_dof_lower_limits[self.actuated_dof_indices], self.arm_dof_upper_limits[self.actuated_dof_indices])

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        self._arms.set_joint_position_targets(
            self.cur_targets[:, self.actuated_dof_indices], indices=None, joint_indices=self.actuated_dof_indices
        )

        #print(object_dist)
        # Apply the gripper action
        
        gripper_actions = self.actions[:, 5].to(dtype=torch.int64)
        for i, action in enumerate(gripper_actions):
            if object_dist[i] < 0.1:
                if action.item() == 0: # Open the gripper
                    #print('open gripper')
                    self._ur10._gripper.open()
                else: # Close the gripper
                    #print('close gripper')
                    self._ur10._gripper.close()

        if self._task_cfg['sim2real']['enabled'] and self.test and self.num_envs == 1:
            # Only retrieve the 0-th joint position even when multiple envs are used
            cur_joint_pos = self._arms.get_joint_positions(indices=[0], joint_indices=self.actuated_dof_indices)
            # Send the current joint positions to the real robot
            joint_pos = cur_joint_pos[0]
            if torch.any(joint_pos < self.arm_dof_lower_limits) or torch.any(joint_pos > self.arm_dof_upper_limits):
                print("get_joint_positions out of bound, send_joint_pos skipped")
            else:
                self.send_joint_pos(joint_pos)


    def is_done(self):
        pass

    def reset_target_pose(self, env_ids):
        # reset goal and object
        indices = env_ids.to(dtype=torch.int32)

        # Random Postions
        object_rand_floats = torch_rand_float(1.0, 1.0, (len(env_ids), 4), device=self.device)
        goal_rand_floats = torch_rand_float(-1.0, -1.0, (len(env_ids), 4), device=self.device)
        new_pos = self.get_reset_target_new_pos(len(env_ids))
        new_pos_object = self.get_reset_target_new_pos(len(env_ids))
        new_rot = randomize_rotation(goal_rand_floats[:, 0], goal_rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        new_rot_object = randomize_rotation(object_rand_floats[:, 0], object_rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        
        #Static Positions, for testing
        # value = torch.tensor([0.8, 0.0, 0.1], device=self.device)
        # value_object = torch.tensor([-0.8, 0.0, 0.1], device=self.device)
        # new_pos = value.repeat(len(env_ids), 1)
        # new_pos_object = value_object.repeat(len(env_ids), 1)

        # print('new_pos_goal: ' + str(new_pos))
        # print('new_pos_object: '+ str(new_pos_object))

        self.object_pos[env_ids] = new_pos_object
        self.object_rot[env_ids] = new_rot_object

        object_pos, object_rot = self.object_pos.clone(), self.object_rot.clone()
        object_pos[env_ids] = self.object_pos[env_ids] + self._env_pos[env_ids] # add world env pos

        self._objects.set_world_poses(object_pos[env_ids], object_rot[env_ids], indices)
        #self.reset_object_buf[env_ids] = 0

        self.goal_pos[env_ids] = new_pos
        self.goal_rot[env_ids] = new_rot

        goal_pos, goal_rot = self.goal_pos.clone(), self.goal_rot.clone()
        goal_pos[env_ids] = self.goal_pos[env_ids] + self._env_pos[env_ids] # add world env pos

        self._goals.set_world_poses(goal_pos[env_ids], goal_rot[env_ids], indices)
        self.reset_goal_buf[env_ids] = 0

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_arm_dofs * 2 + 5), device=self.device)

        self.reset_target_pose(env_ids)

        # reset arm
        delta_max = self.arm_dof_upper_limits - self.arm_dof_default_pos
        delta_min = self.arm_dof_lower_limits - self.arm_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * (rand_floats[:, 5:5+self.num_arm_dofs] + 1.0) * 0.5

        pos = self.arm_dof_default_pos + self.reset_dof_pos_noise * rand_delta
        dof_pos = torch.zeros((self.num_envs, self._arms.num_dof), device=self.device)
        dof_pos[env_ids, :self.num_arm_dofs] = pos

        dof_vel = torch.zeros((self.num_envs, self._arms.num_dof), device=self.device)
        dof_vel[env_ids, :self.num_arm_dofs] = self.arm_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_arm_dofs:5+self.num_arm_dofs*2]

        self.prev_targets[env_ids, :self.num_arm_dofs] = pos
        self.cur_targets[env_ids, :self.num_arm_dofs] = pos
        self.arm_dof_targets[env_ids, :self.num_arm_dofs] = pos

        self._arms.set_joint_position_targets(self.arm_dof_targets[env_ids], indices)
        # set_joint_positions doesn't seem to apply immediately.
        self._arms.set_joint_positions(dof_pos[env_ids], indices)
        self._arms.set_joint_velocities(dof_vel[env_ids], indices)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def compute_arm_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot,
    end_effector_pos,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float
):  
    #print(end_effector_pos)
    #goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    goal_dist = torch.norm(end_effector_pos - object_pos, p=2, dim=-1)
    
    # use the L1 norm, this line calculates the distance between the end effector and the object
    # In this line, torch.abs(end_effector_pos - object_pos) computes the absolute difference between the end effector position and the object position along each dimension. torch.sum(..., dim=-1) then adds up these absolute differences to compute the Manhattan distance.
    # goal_dist = torch.sum(torch.abs(end_effector_pos - object_pos), dim=-1)


    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0)) # changed quat convention

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # large positive reward for picking up the object
    # pickup_reward = (goal_dist < 0.1).float() * 100.0
    # dist_rew += pickup_reward
    
    # Compute the success reward (if the end effector is close enough to the object)
    success_reward = (goal_dist < success_tolerance).float() * reach_goal_bonus
    
    # large positive reward for lifting the object off the ground
    # lift_reward = (object_pos[:, 2] > 0.5).float() * 50.0


    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + action_penalty * action_penalty_scale + rot_rew + success_reward
    

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(goal_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)
    print('reward: ' + str(reward))

    resets = reset_buf
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    # place holders restets, goal_resets and cons_successes
    # resets = torch.zeros_like(reset_buf)
    # goal_resets = torch.zeros_like(reset_goal_buf)
    # cons_successes = torch.zeros_like(consecutive_successes)
    
    return reward, resets, goal_resets, progress_buf, successes, cons_successes
