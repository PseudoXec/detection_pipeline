# Person event API: what the Pi sends to the command center

For the command-center (C#) team. The Pi detects people and finds a face in each one. It does **not** identify
anyone and has no matching code at all: the command center receives the face image and its landmarks and does
alignment, embedding and matching on the server.

## Request

`POST <api.person_endpoint_url>` with `Content-Type: multipart/form-data`. One request per person track.
All metadata is in form fields (nothing in the query string); the images are file parts.

| File part | Always? | Content |
|---|---|---|
| `PersonImage` | yes | JPEG crop of the person |
| `FaceImage` | only when `FaceDetected = true` | JPEG crop of the face (the detected face plus a 20% margin) |

| Form field | Type | Sent when | Meaning |
|---|---|---|---|
| `EventId` | uuid | always | **Idempotency key.** Store it; if you see it again, answer 409 |
| `DeviceId` | string | always | Which Pi sent it |
| `SessionId` | string | always | One program run on the Pi (`TrackId` restarts every run) |
| `TrackId` | string | always | The person's tracker id, e.g. `p12`. Unique only together with `DeviceId` + `SessionId` |
| `CameraId` | int | when `api.camera_id` is set | The command center's id for this camera |
| `CameraName`, `CameraIpAddress`, `CameraLocation` | string | when the camera reports them | Same values the vehicle endpoint receives |
| `CameraSource` | string | always | Stream address **with the login removed** |
| `DetectedAt` | ISO 8601 with offset | always | When the frame was captured, Pi local time, e.g. `2026-10-05T14:30:15+08:00` |
| `FrameWidth`, `FrameHeight` | int | always | Size of the full camera frame the person box refers to |
| `PersonConfidence` | float 0..1 | always | Person detector confidence |
| `PersonBoxX1`, `PersonBoxY1`, `PersonBoxX2`, `PersonBoxY2` | float | always | Person box in **full-frame pixels** (top-left and bottom-right corners) |
| `FaceDetected` | `true` / `false` | always | Whether a usable face was found |
| `FaceConfidence` | float 0..1 | face found | Face detector confidence |
| `FaceBoxX1`, `FaceBoxY1`, `FaceBoxX2`, `FaceBoxY2` | float | face found | Face box in **pixels of `PersonImage`** |
| `FaceLandmarks` | JSON text | face found | `[[x,y],[x,y],[x,y],[x,y],[x,y]]`, five points in **pixels of `PersonImage`** |
| `FaceQualityScore` | float 0..1 | face found | Size, sharpness and detector confidence combined; higher is better |
| `PersonDetectMs`, `PersonCropMs`, `FaceDetectMs`, `FaceCropMs`, `TotalPipelineMs` | float (ms) | when measured | Timings on the Pi. `TotalPipelineMs` is frame captured to event built |

A person whose face was never found is still sent, with `FaceDetected = false`, no `FaceImage` and none of the face fields.

### Coordinates and landmark order

* `PersonBox*` is in the full frame. `FaceBox*` and `FaceLandmarks` are in the pixel space of **`PersonImage`**, not of the full frame and **not of `FaceImage`**.
* To align a face, use `PersonImage` + `FaceLandmarks`. (`FaceImage` is a convenience copy for previews; its crop offset is not sent.)
* Landmark order is the order OpenCV's YuNet face detector returns: right eye, left eye, nose tip, right mouth corner, left mouth corner (OpenCV's own naming). Check "right" and "left" against one real sample before relying on it.

### What the server does with it

1. Decode `PersonImage`; align the face from the five `FaceLandmarks` to your recognition model's template.
2. Embed and match against the registered people.
3. Optionally ignore events with a low `FaceQualityScore` before spending time on them.

## Answers and what the Pi does next

| Your answer | The Pi |
|---|---|
| 200 / 201 / 202 / 204 | Delivered; the row is removed (or marked synced) |
| **409** | "Already have this `EventId`": treated as delivered |
| 400, 413, 415, 422 | This event is unacceptable: parked locally (`synced = 2`) and never retried |
| Anything else (401, 403, 404, 408, 429, 5xx, no connection) | Kept and retried: every 60 s at first, then less often per event (x2 each failure, up to 1 hour). **Events are never given up on**, so an outage of any length loses nothing |

The response body is ignored. A 401/403/404 means a wrong token or URL and is retried on purpose, so fixing the setup releases the whole backlog.

## Example

```bash
curl -X POST "http://<command-center>/api/AiDetection/PersonDetected" \
  -F EventId=0b1f1c52-2c2e-4d21-9a53-6d7a4c7a9f10 -F DeviceId=pi-01 -F SessionId=3f9c... -F TrackId=p12 \
  -F CameraId=4 -F CameraName=Gate -F CameraSource=rtsp://192.168.100.229:554/stream1 \
  -F DetectedAt=2026-10-05T14:30:15+08:00 -F FrameWidth=1280 -F FrameHeight=720 \
  -F PersonConfidence=0.9123 -F PersonBoxX1=580.0 -F PersonBoxY1=210.0 -F PersonBoxX2=700.0 -F PersonBoxY2=510.0 \
  -F FaceDetected=true -F FaceConfidence=0.9512 -F FaceBoxX1=40.0 -F FaceBoxY1=10.0 -F FaceBoxX2=80.0 -F FaceBoxY2=60.0 \
  -F 'FaceLandmarks=[[50,25],[70,25],[60,35],[52,48],[68,48]]' -F FaceQualityScore=0.81 \
  -F TotalPipelineMs=77.0 \
  -F PersonImage=@person.jpg -F FaceImage=@face.jpg
```

## C# model (a starting point)

```csharp
public class PersonDetectionRequest
{
    public Guid EventId { get; set; }
    public string DeviceId { get; set; } = "";
    public string SessionId { get; set; } = "";
    public string TrackId { get; set; } = "";
    public int? CameraId { get; set; }
    public string? CameraName { get; set; }
    public string? CameraIpAddress { get; set; }
    public string? CameraLocation { get; set; }
    public string CameraSource { get; set; } = "";
    public DateTimeOffset DetectedAt { get; set; }
    public int FrameWidth { get; set; }
    public int FrameHeight { get; set; }

    public double PersonConfidence { get; set; }
    public double PersonBoxX1 { get; set; }
    public double PersonBoxY1 { get; set; }
    public double PersonBoxX2 { get; set; }
    public double PersonBoxY2 { get; set; }

    public bool FaceDetected { get; set; }
    public double? FaceConfidence { get; set; }
    public double? FaceBoxX1 { get; set; }
    public double? FaceBoxY1 { get; set; }
    public double? FaceBoxX2 { get; set; }
    public double? FaceBoxY2 { get; set; }
    public string? FaceLandmarks { get; set; }        // JSON: [[x,y] x 5]
    public double? FaceQualityScore { get; set; }

    public double? PersonDetectMs { get; set; }
    public double? PersonCropMs { get; set; }
    public double? FaceDetectMs { get; set; }
    public double? FaceCropMs { get; set; }
    public double? TotalPipelineMs { get; set; }

    public IFormFile PersonImage { get; set; } = default!;
    public IFormFile? FaceImage { get; set; }
}

[HttpPost("api/AiDetection/PersonDetected")]
public async Task<IActionResult> PersonDetected([FromForm] PersonDetectionRequest request)
{
    if (await _events.ExistsAsync(request.EventId))
        return Conflict();                 // 409: the Pi treats this as delivered
    await _events.SaveAsync(request);      // then align + embed + match on the server
    return Ok();
}
```

## What stays on the Pi

The Pi's SQLite row also keeps a few diagnostics that are **not sent**: first/last time the person was seen,
face sharpness, number of face attempts, model names, pipeline version and CPU temperature. Add any of them to
`build_form()` in `api/person_client.py` if you want them on the server.
