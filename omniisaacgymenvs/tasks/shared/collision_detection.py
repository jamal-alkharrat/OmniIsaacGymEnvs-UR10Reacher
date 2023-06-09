def subscribe_to_contact(self):
        # apply contact report
        ### This would be an example of each object managing their own collision
        self._contact_report_sub = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact_report_event)
        contactReportAPI = PhysxSchema.PhysxContactReportAPI.Apply(self.prim)
        contactReportAPI.CreateThresholdAttr().Set(self.contact_thresh)

def _on_contact_report_event(self, contact_headers, contact_data):

    # Check if a collision was because of a player
    for contact_header in contact_headers:          
        collider_1 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0))
        collider_2 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1))

        contacts = [collider_1, collider_2]

        if  self.player_name in contacts and self.prim_path in contacts:
            print('Hiding')
            attribute = self.prim.GetAttribute("visibility")
            attribute.Set("invisible")
            # Turn of its collision property
            collision_attr = self.prim.GetAttribute("physics:collisionEnabled")
            collision_attr.Set(False)

import omni.kit.commands
from pxr import UsdGeom, Gf
async def create_scenario(self):


        self.sensor_offset= Gf.Vec3d(40,0,0)
        self.color = (1, 0, 0, 1)


       
        result, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateContactSensor",
            path="/sensor",
            parent="ur10/wrist_2_link/ur10_wrist_2",
            min_threshold=0,
            max_threshold=10000000,
            color=self.color,
            radius=0.12,
            sensor_period=-1,
            translation=self.sensor_offsets,
            visualize=True,)
            

        self._events = omni.usd.get_context().get_stage_event_stream()
        self._stage_event_subscription = self._events.create_subscription_to_pop(
            self._on_stage_event, name="Contact Sensor Sample stage Watch"
        )