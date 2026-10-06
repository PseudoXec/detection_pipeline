"""Work out which camera to read and what it is called, before anything starts."""
import logging

from api.camera_source_client import fetch_camera_source
from camera.isapi_client import CameraDeviceInfo, extract_credentials, extract_host, fetch_device_info
from config.config import PipelineConfig

log = logging.getLogger("pipeline")


def resolve_camera_source(config: PipelineConfig) -> None:
    """With camera.source_mode "api" the command center decides the RTSP URL."""
    if config.camera.source_mode != "api":
        return
    if config.camera.image or config.camera.folder:
        return

    source = fetch_camera_source(
        config.api.roi_endpoint_url,
        config.api.camera_id,
        config.api.roi_fetch_timeout_seconds,
    )
    if source is None:
        if config.camera.rtsp_url:
            log.warning("[camera-source] falling back to the static camera.rtsp_url from camera/config.yaml")
        return

    config.camera.rtsp_url = source.rtsp_url


def resolve_camera_info(config: PipelineConfig):
    """Camera name / IP / location (Hikvision ISAPI), stamped on every stored event."""
    rtsp_url = config.camera.rtsp_url
    if not rtsp_url:
        return None

    host = extract_host(rtsp_url)
    if not config.camera.isapi_enabled or not host:
        return CameraDeviceInfo(ip_address=host)

    username = config.camera.isapi_username
    password = config.camera.isapi_password
    if not username:
        username, password = extract_credentials(rtsp_url)

    return fetch_device_info(
        host,
        port=config.camera.isapi_port,
        username=username,
        password=password,
        use_https=config.camera.isapi_https,
        timeout_seconds=config.camera.isapi_timeout_seconds,
        channel_id=config.camera.isapi_channel_id,
    )
