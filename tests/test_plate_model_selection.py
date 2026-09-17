import argparse
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import main
import ocr_cropped_plates as ocr


class PlateModelSelectionTest(unittest.TestCase):
    def test_process_single_image_uses_separate_plate_model(self):
        args = argparse.Namespace(
            vehicle_classes=None,
            vehicle_conf_threshold=0.35,
            vehicle_iou_threshold=0.45,
            imgsz=None,
            plate_conf_threshold=0.40,
            plate_containment_threshold=0.8,
            plate_min_iou=0.1,
            vehicle_crop_padding=0.30,
            vehicle_crop_min_height=200,
            plate_crop_padding=8,
            plate_crop_min_height=120,
            min_confidence=0.80,
        )

        vehicle_model = object()
        plate_model = object()
        seen = []

        def fake_run_local_detection(model, image_path, conf_threshold, iou_threshold, imgsz, classes):
            seen.append((model, classes, image_path))
            if model is vehicle_model:
                return [{
                    "x": 50.0, "y": 50.0, "width": 40.0, "height": 40.0,
                    "class": "car", "confidence": 0.9,
                }]
            return []

        with patch.object(main.detect, "run_local_detection", side_effect=fake_run_local_detection), \
             patch.object(main.cv2, "imread", return_value=np.zeros((100, 100, 3), dtype=np.uint8)), \
             patch.object(main.cv2, "imwrite", return_value=True), \
             patch.object(main, "match_plates_to_vehicles", return_value=[]), \
             patch.object(main, "run_ocr_on_plate_crops", return_value=[]):
            main.process_single_image(
                image_path="sample.jpg",
                args=args,
                vehicle_model=vehicle_model,
                plate_model=plate_model,
                ocr_engine=None,
                date_dir="out",
                vehicle_dir="out/vehicle_detection",
                plate_dir="out/plate_detection",
                ocr_dir="out/ocr",
                work_dir="out/work",
                agg_rows=[],
            )

        self.assertEqual(seen[0][0], vehicle_model)
        self.assertEqual(seen[1][0], plate_model)
        self.assertIsNone(seen[0][1])
        self.assertIsNone(seen[1][1])
        self.assertEqual(seen[0][2], "sample.jpg")
        self.assertNotEqual(seen[1][2], "sample.jpg")

    def test_plate_text_must_be_english_alphanumeric_only(self):
        self.assertTrue(ocr.is_valid_plate_text("ABC1234"))
        self.assertTrue(ocr.is_valid_plate_text("abc1234"))
        self.assertTrue(ocr.is_valid_plate_text("83-6852"))
        self.assertTrue(ocr.is_valid_plate_text("LAB_4669"))
        self.assertFalse(ocr.is_valid_plate_text("城"))
        self.assertFalse(ocr.is_valid_plate_text("J8S B"))
        self.assertFalse(ocr.is_valid_plate_text("A"))

    def test_draw_detection_preview_can_be_disabled(self):
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        vehicle = [{"x": 10, "y": 10, "width": 20, "height": 20, "confidence": 0.9}]

        result = main.draw_detection_preview(frame, vehicle, [], draw_boxes=False)

        np.testing.assert_array_equal(result, frame)

    def test_should_reprocess_stream_tracks_only_for_new_or_incomplete_tracks(self):
        track_state = {
            "stream_v1": {"ocr_done": True, "ocr_pending": False, "finalized": False},
            "stream_v2": {"ocr_done": False, "ocr_pending": False, "finalized": False},
        }

        self.assertTrue(main.should_reprocess_stream_tracks([
            {"track_id": "stream_v1"},
            {"track_id": "stream_v2"},
        ], track_state))

        self.assertTrue(main.should_reprocess_stream_tracks([
            {"track_id": "stream_v3"},
        ], track_state))

        self.assertFalse(main.should_reprocess_stream_tracks([
            {"track_id": "stream_v1"},
        ], {"stream_v1": {"ocr_done": True, "ocr_pending": False, "finalized": False}}))

    def test_rtsp_stream_skips_frames_and_releases_capture(self):
        class FakeCapture:
            def __init__(self):
                self.frames = [
                    (True, np.zeros((20, 20, 3), dtype=np.uint8)),
                    (True, np.zeros((20, 20, 3), dtype=np.uint8)),
                    (True, np.zeros((20, 20, 3), dtype=np.uint8)),
                ]
                self.read_count = 0
                self.released = False

            def isOpened(self):
                return not self.released

            def set(self, property_id, value):
                return True

            def read(self):
                if self.read_count >= len(self.frames):
                    return False, None
                frame = self.frames[self.read_count]
                self.read_count += 1
                return frame

            def release(self):
                self.released = True

        capture = FakeCapture()
        args = argparse.Namespace(
            stream_frame_skip=2,
            stream_reconnect_delay=0,
            stream_max_frames=2,
        )
        processed_paths = []

        def fake_process_single_image(image_path, *unused_args):
            processed_paths.append(image_path)
            return 1

        with patch.object(main.cv2, "imwrite", return_value=True), \
               patch.object(main, "process_single_image", side_effect=fake_process_single_image), \
               patch.object(main.detect, "run_local_tracking", return_value=[]):
            total = main.process_rtsp_stream(
                "rtsp://camera/stream", args, object(), object(), object(),
                "out", "out/vehicle", "out/plate", "out/ocr", tempfile.gettempdir(), [],
                capture_factory=lambda url: capture,
            )

        self.assertGreaterEqual(total, 1)
        self.assertGreaterEqual(len(processed_paths), 1)
        self.assertEqual(capture.read_count, 3)
        self.assertTrue(capture.released)

    def test_stream_tracks_reuse_id_for_overlapping_vehicle(self):
        tracker = {"tracks": {}, "next_track_number": 1}
        first = {"x": 50, "y": 50, "width": 40, "height": 40, "confidence": 0.9}
        second = {"x": 52, "y": 51, "width": 40, "height": 40, "confidence": 0.9}

        main.update_stream_tracks([first], tracker)
        main.update_stream_tracks([second], tracker)

        self.assertEqual(first["track_id"], "stream_v1")
        self.assertEqual(second["track_id"], "stream_v1")
        self.assertEqual(tracker["next_track_number"], 2)

    def test_should_trigger_tracking_respects_cooldown_for_stable_tracks(self):
        completed_track = {
            "stream_v1": {"ocr_done": True, "finalized": False, "ocr_pending": False, "last_seen_frame": 2},
        }
        incomplete_track = {
            "stream_v2": {"ocr_done": False, "finalized": False, "ocr_pending": False, "last_seen_frame": 2},
        }

        self.assertFalse(main.should_trigger_tracking_for_frame(
            frame_number=3,
            last_tracking_frame=1,
            stream_track_state=completed_track,
            cooldown_frames=2,
        ))
        self.assertTrue(main.should_trigger_tracking_for_frame(
            frame_number=5,
            last_tracking_frame=1,
            stream_track_state=incomplete_track,
            cooldown_frames=2,
        ))


if __name__ == "__main__":
    unittest.main()
