from __future__ import annotations

import json
from typing import Any

from common.schemas import TargetCommand


class FlightBridge:
    """
    Minimal bridge to a flight stack.
    Replace `send` internals with MAVSDK/MAVROS/PX4 integration when ready.
    """

    def send(self, command: TargetCommand) -> None:
        payload: dict[str, Any] = command.model_dump()
        print(f"[flight_bridge] sending command: {json.dumps(payload, ensure_ascii=False)}")

    def send_obj(self, command: dict[str, Any] | TargetCommand) -> None:
        if isinstance(command, TargetCommand):
            parsed = command
        else:
            parsed = TargetCommand.model_validate(command)
        self.send(parsed)
