import pybullet as p
import os

p.connect(p.GUI)

# until body 7
for body in ["body_1.obj", "body_2.obj", "body_3.obj", "body_4.obj", "body_5.obj", "body_6.obj", "body_7_on.obj"]:
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=body,
        meshScale=[1, 1, 1],
        rgbaColor=[1, 1, 1, 1]  # Let texture show
    )

    p.createMultiBody(baseVisualShapeIndex=visual_shape_id)

input("Press Enter to exit...")
p.disconnect()