from __future__ import annotations

from common.schemas import DronePose


class ViconBridge:
    """
    Stub adapter for an existing Vicon pipeline.
    Wire this class to your real Vicon source and return current drone pose in meters/radians.
    """

    def get_pose(self) -> DronePose:
        return DronePose(x=0.0, y=0.0, z=1.0, yaw=0.0)
