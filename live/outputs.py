"""Builds the live outputs: the push publisher and the HTTP server (live view + mode-switch endpoint)."""
from config.config import PipelineConfig
from live.box_publisher import LiveBoxPublisher
from live.live_server import LiveServer
from live.sinks import LiveSinks


def build_live_outputs(config: PipelineConfig, roi, manager, sinks: LiveSinks):
    """The live view (frames + boxes) and the command-center control endpoint share one HTTP server."""
    publisher = None
    if config.switches.live_boxes:
        if config.live.endpoint_url:
            publisher = LiveBoxPublisher(
                endpoint_url=config.live.endpoint_url, camera_id=config.live.camera_id,
                max_hz=config.live.max_hz, timeout_seconds=config.live.timeout_seconds,
            ).start()
        else:
            print("[pipeline] switches.live_boxes is on but live.endpoint_url is not set (live/config.yaml) - live feed disabled")

    server = None
    if config.switches.live_stream or config.mode.control_enabled:
        control = config.mode.control_enabled
        server = LiveServer(
            camera_id=config.live.camera_id, host=config.live.serve_host, port=config.live.serve_port,
            stream_fps=config.live.stream_fps, stream_width=config.live.stream_width,
            jpeg_quality=config.live.jpeg_quality, box_max_age_seconds=config.live.box_max_age_seconds,
            auth_token=config.live.auth_token, box_visual_tracking=config.live.box_visual_tracking,
            box_extrapolate=config.live.box_extrapolate,
            box_extrapolate_max_seconds=config.live.box_extrapolate_max_seconds,
            roi_get=roi.get_polygon, roi_set=roi.set_polygon,
            mode_status=manager.status if control else None,
            mode_request=manager.request if control else None,
            mode_choices=manager.choices if control else None,
            live_enabled=config.switches.live_stream,
        )
    sinks.publisher, sinks.server = publisher, server
    return publisher, server
