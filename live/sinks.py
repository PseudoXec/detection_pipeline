"""One place the detection modes publish boxes to, whichever live outputs are switched on."""
from typing import Any, Dict, List, Optional, Set


class LiveSinks:
    """Fan-out to the box publisher (push) and the live server (pull). Either may be None."""

    def __init__(self, publisher: Optional[Any] = None, server: Optional[Any] = None):
        self.publisher = publisher
        self.server = server

    def publish(
        self,
        frame_shape: tuple,
        predictions: List[Dict[str, Any]],
        in_roi_ids: Set[int],
        captured_at: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> None:
        for sink in (self.publisher, self.server):
            if sink is not None:
                sink.publish(frame_shape, predictions, in_roi_ids, captured_at, mode)

    def clear(self) -> None:
        """Forget all boxes (the detection mode just changed)."""
        if self.server is not None:
            self.server.clear_boxes()
        if self.publisher is not None:
            self.publisher.clear()
