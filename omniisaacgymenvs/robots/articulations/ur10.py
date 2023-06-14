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


from typing import Optional
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.manipulators.grippers.surface_gripper import SurfaceGripper
from omni.isaac.core.utils.prims import get_prim_at_path

import carb

#### Added from William Rodmann
from omni.isaac.motion_planning import _motion_planning
import time
import numpy as np
from pxr import Usd, UsdGeom, Gf, UsdPhysics, PhysxSchema

class EndEffector:
    def __init__(self, dc, mp, ar, rmp_handle):
        self.dc = dc
        self.ar = ar
        self.mp = mp
        self.rmp_handle = rmp_handle
        self.gripper = None
        #self.status = Status(mp, rmp_handle)
        self.UpRot = Gf.Rotation(Gf.Vec3d(0, 0, 1), 90)

    def go_local(
           self,
        target=None,
        orig=[],
        axis_x=[],
        axis_y=[],
        axis_z=[],
        required_orig_err=0.01,
        required_axis_x_err=0.01,
        required_axis_y_err=0.01,
        required_axis_z_err=0.01,
        orig_thresh=None,
        axis_x_thresh=None,
        axis_y_thresh=None,
        axis_z_thresh=None,
        approach_direction=[],
        approach_standoff=0.1,
        approach_standoff_std_dev=0.001,
        use_level_surface_orientation=False,
        use_target_weight_override=True,
        use_default_config=False,
        wait_for_target=True,
        wait_time=None, 
    ):
        self.target_weight_override_value = 10000.0
        self.target_weight_override_std_dev = 0.03
        # TODO: handle all errors?
        if orig_thresh:
            required_orig_err = orig_thresh
        if axis_x_thresh:
            required_axis_x_err = axis_x_thresh
        if axis_y_thresh:
            required_axis_y_err = axis_y_thresh
        if axis_z_thresh:
            required_axis_z_err = axis_z_thresh

        if target:
            orig = target["orig"]
            if "axis_x" in target and target["axis_x"] is not None:
                axis_x = target["axis_x"]
            if "axis_y" in target and target["axis_y"] is not None:
                axis_y = target["axis_y"]
            if "axis_z" in target and target["axis_z"] is not None:
                axis_z = target["axis_z"]

        orig = np.array(orig)
        axis_x = np.array(axis_x)
        axis_y = np.array(axis_y)
        axis_z = np.array(axis_z)
        approach = _motion_planning.Approach((0, 0, -1), 0, 0)

        if len(approach_direction) != 0:
            approach = _motion_planning.Approach(approach_direction, approach_standoff, approach_standoff_std_dev)

        pose_command = _motion_planning.PartialPoseCommand()
        if len(orig) > 0:
            pose_command.set(_motion_planning.Command(orig, approach), int(_motion_planning.FrameElement.ORIG))
        if len(axis_x) > 0:
            pose_command.set(_motion_planning.Command(axis_x), int(_motion_planning.FrameElement.AXIS_X))
        if len(axis_y) > 0:
            pose_command.set(_motion_planning.Command(axis_y), int(_motion_planning.FrameElement.AXIS_Y))
        if len(axis_z) > 0:
            pose_command.set(_motion_planning.Command(axis_z), int(_motion_planning.FrameElement.AXIS_Z))

        self.mp.goLocal(self.rmp_handle, pose_command)

        if wait_for_target and wait_time:
            error = 1
            future_time = time.time() + wait_time

            while error > required_orig_err and time.time() < future_time:
                time.sleep(0.1)
                error = self.mp.getError(self.rmp_handle)

####

class UR10(Robot):
    def __init__(
        self,
        prim_path: str, #  prim_path=self.default_zero_env_path + "/ur10"
        name: Optional[str] = "UR10",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
        ####################
        end_effector_prim_name: Optional[str] = None,
        attach_gripper: bool = False,
        gripper_usd: Optional[str] = "default",
        ####################
    ) -> None:

        self._usd_path = usd_path
        self._name = name

        ####################
        self._end_effector = EndEffector(self.dc, self.mp, self.ar, self.rmp_handle)
        self._gripper = None
        self._end_effector_prim_name = end_effector_prim_name
        ####################

        if self._usd_path is None:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
            #self._usd_path = "omniverse://localhost/Projects/J3soon/Isaac/2022.1/Isaac/Robots/UR10/ur10_instanceable.usd"
            self._usd_path = "/home/willi/Dokumente/Omniverse-Pick-and-Place/OmniIsaacGymEnvs-UR10Reacher/Isaac/2022.1/Isaac/Robots/UR10/ur10_short_suction_instanceable.usd"
        
        ####################
        if self._end_effector_prim_name is None: #default --> Attaches the gripper
            self._end_effector_prim_path = prim_path + "/ee_link"
        else:
            self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name
        ####################
        
        # Depends on your real robot setup
        self._position = torch.tensor([0.0, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([1.0, 0.0, 0.0, 0.0]) if orientation is None else orientation

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )
        ####################
        self._gripper_usd = gripper_usd #default
        
        print('gripper_usd: ' + str(gripper_usd))
        print('attach_gripper: '+ str(attach_gripper) )
        if attach_gripper:
            if gripper_usd == "default":
                assets_root_path = get_assets_root_path()
                if assets_root_path is None:
                    carb.log_error("Could not find Isaac Sim assets folder")
                    return
                gripper_usd = assets_root_path + "/Isaac/Robots/UR10/Props/short_gripper.usd"
                add_reference_to_stage(usd_path=gripper_usd, prim_path=self._end_effector_prim_path)
                self._gripper = SurfaceGripper(
                    end_effector_prim_path=self._end_effector_prim_path, translate=0.1611, direction="x"
                )
            elif gripper_usd is None:
                print('Not adding a gripper usd, the gripper already exists in the ur10 asset')
                carb.log_warn("Not adding a gripper usd, the gripper already exists in the ur10 asset")
                self._gripper = SurfaceGripper(
                    end_effector_prim_path=self._end_effector_prim_path, translate=0.1611, direction="x"
                )
            else:
                print('NotImplementedError')
                raise NotImplementedError
        self._attach_gripper = attach_gripper
        return
        ####################

    @property
    def attach_gripper(self) -> bool:
        """[summary]

        Returns:
            bool: [description]
        """
        return self._attach_gripper

    @property
    def end_effector(self) -> RigidPrim:
        """[summary]

        Returns:
            RigidPrim: [description]
        """
        return self._end_effector

    @property
    def gripper(self) -> SurfaceGripper:
        """[summary]

        Returns:
            SurfaceGripper: [description]
        """
        return self._gripper

    def initialize(self, physics_sim_view=None) -> None:
        """[summary]
        """
        super().initialize(physics_sim_view)
        if self._attach_gripper:
            self._gripper.initialize(physics_sim_view=physics_sim_view, articulation_num_dofs=self.num_dof)
        self._end_effector = RigidPrim(prim_path=self._end_effector_prim_path, name=self.name + "_end_effector")
        self.disable_gravity()
        self._end_effector.initialize(physics_sim_view)
        print("Attached gripper is initialized")
        return

    def post_reset(self) -> None:
        """[summary]
        """
        Robot.post_reset(self)
        self._end_effector.post_reset()
        self._gripper.post_reset()
        return
