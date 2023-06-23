# Platform dev


```python
# Define the force to be applied. This should be a 3-element tensor representing the x, y, and z components of the force.
# For example, to apply an upward force of 9.81 * mass to counteract gravity:
masses = self._platforms.get_masses()
print('masses: ', masses) # --> 3.6
force = torch.tensor([0.0, 0.0, 3.6 * 9.81]) # Mass * Gravity
self._platforms.apply_forces(forces=force, indices=indices) # trying to disable gravity for the platform
self._platforms.disable_rigid_body_physics() # trying to disable gravity for the platform
```