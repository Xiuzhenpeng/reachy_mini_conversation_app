"""Tests for utility helpers."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from reachy_mini_conversation_app.utils import (
    CameraVisionInitializationError,
    initialize_camera_and_vision,
)


def test_initialize_camera_and_vision_raises_when_head_tracker_init_fails() -> None:
    """Head-tracker startup failures should be reported through the clean init error path."""
    args = argparse.Namespace(
        no_camera=False,
        head_tracker="mediapipe",
    )

    with (
        patch("reachy_mini_conversation_app.utils.CameraWorker") as mock_camera_worker,
        patch(
            "reachy_mini_conversation_app.vision.head_tracking.mediapipe.MediapipeHeadTracker",
            side_effect=RuntimeError("tracker init failed"),
        ),
    ):
        with pytest.raises(
            CameraVisionInitializationError,
            match="Failed to initialize mediapipe head tracker: tracker init failed",
        ):
            initialize_camera_and_vision(args, MagicMock())

    mock_camera_worker.assert_not_called()


def test_initialize_camera_and_vision_uses_mediapipe_head_tracker_in_process() -> None:
    """MediaPipe head tracking should use the in-process toolbox tracker."""
    args = argparse.Namespace(
        no_camera=False,
        head_tracker="mediapipe",
    )

    current_robot = MagicMock()
    mediapipe_head_tracker = MagicMock()
    with (
        patch("reachy_mini_conversation_app.utils.CameraWorker") as mock_camera_worker,
        patch(
            "reachy_mini_conversation_app.vision.head_tracking.mediapipe.MediapipeHeadTracker",
            return_value=mediapipe_head_tracker,
        ),
    ):
        initialize_camera_and_vision(args, current_robot)

    mock_camera_worker.assert_called_once_with(current_robot, mediapipe_head_tracker)
