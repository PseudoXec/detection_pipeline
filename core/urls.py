import re

_CREDENTIALS = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.\-]*://)[^/]*@")


def redact_credentials(url):
    """rtsp://user:password@host:554/path -> rtsp://host:554/path. Anything without a login is returned as is."""
    if not url:
        return url
    return _CREDENTIALS.sub(r"\1", url, count=1)
