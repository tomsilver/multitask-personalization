import pybullet as p
import os

p.connect(p.GUI)

for body in ["chair_base.obj"]:
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=body,
        meshScale=[0.1, 0.1, 0.1],
        rgbaColor=[0.682476, 0.408966, 0.026520, 1]  # Let texture show
    )

    p.createMultiBody(baseVisualShapeIndex=visual_shape_id)

input("Press Enter to exit...")
p.disconnect()