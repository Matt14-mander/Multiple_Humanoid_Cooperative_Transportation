# Airbot model assets

This directory stores Airbot arm and end-effector assets independently from
the combined TRON1 URDFs.

## Current asset

`urdf/airbot_play_with_gripper.urdf` is a byte-for-byte staged copy of the
downloaded `airbot_play_v2_1_with_gripper.urdf`.

The current URDF defines the arm links `base_link` through `link6` and six
actuated arm joints. It does not define independent gripper links or gripper
joints. This is compatible with a fixed gripper representation embedded in
the terminal `link6` mesh: the gripper is visible, but its opening and closing
are not represented as Pinocchio joint coordinates. The URDF also references
mesh files using the original absolute Linux path
`/home/george/Downloads/airbot_play_v6_1/meshes`.

The `meshes/` directory is intentionally reserved for the matching downloaded
mesh set. Do not substitute the combined `airbot_arm.STL` mesh from the TRON1
description: it does not represent the individual links required by this
standalone URDF.

After the matching meshes are available, update the URDF mesh references to
portable paths such as `package://airbot/meshes/link1.STL`, or use a parser
loader that maps the original package root. Only then should this model be
used in MeshCat geometry visualization or merged into a robot model.
