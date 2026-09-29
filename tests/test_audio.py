from pathlib import Path
import subprocess
import pytest

from engine.audio import format_timestamp, get_audio_duration


def test_format_timestamp():
    assert format_timestamp(0.0) == "00:00.000"
    assert format_timestamp(3.82) == "00:03.820"
    assert format_timestamp(65.432) == "01:05.432"
    assert format_timestamp(1503.421) == "25:03.421"
    assert format_timestamp(3665.12) == "01:01:05.120"


def test_get_audio_duration(tmp_path: Path):
    audio_path = tmp_path / "test_tone.wav"
    # Generate a 2.5 second audio tone using ffmpeg
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "sine=frequency=1000:duration=2.5",
        "-c:a", "pcm_s16le",
        str(audio_path)
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    dur = get_audio_duration(audio_path)
    assert pytest.approx(dur, rel=1e-2) == 2.5


def test_verify_rendered_video(tmp_path: Path):
    from engine.audio import DURATION_TOLERANCE, verify_rendered_video

    assert DURATION_TOLERANCE == 2.0

    # Generate a simple 2.0s video with audio
    video_path = tmp_path / "test_verify.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "color=c=blue:s=320x240:d=2.0",
        "-f", "lavfi",
        "-i", "sine=frequency=1000:duration=2.0",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest",
        str(video_path)
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    # Test with default tolerance (DURATION_TOLERANCE = 2.0)
    report = verify_rendered_video(video_path, expected_audio_duration=2.0)
    assert report.video_valid is True
    assert report.width == 320
    assert report.height == 240
    assert report.video_codec == "h264"
    assert report.audio_codec == "aac"
    assert pytest.approx(report.video_duration, abs=0.2) == 2.0

    # Test non-existent video
    missing_report = verify_rendered_video(tmp_path / "non_existent.mp4", 2.0)
    assert missing_report.video_valid is False
    assert len(missing_report.errors) > 0


@pytest.mark.parametrize(
    "audio_dur,video_dur,expected_diff,expected_pass",
    [
        (814.372, 814.000, 0.372, True),
        (814.372, 812.500, 1.872, True),
        (814.372, 812.372, 2.000, True),
        (814.372, 812.371, 2.001, False),
        (814.372, 816.372, 2.000, True),
        (814.372, 816.373, 2.001, False),
    ],
)
def test_verify_rendered_video_duration_tolerance(
    tmp_path: Path, monkeypatch, audio_dur, video_dur, expected_diff, expected_pass
):
    import json
    from engine.audio import DURATION_TOLERANCE, verify_rendered_video

    assert DURATION_TOLERANCE == 2.0

    dummy_video = tmp_path / f"video_{video_dur}.mp4"
    dummy_video.write_bytes(b"dummy mp4 content for testing")

    mock_ffprobe_data = {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30/1",
                "duration": str(video_dur),
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "duration": str(audio_dur),
            },
        ],
        "format": {
            "duration": str(video_dur),
        },
    }

    class MockProcessResult:
        stdout = json.dumps(mock_ffprobe_data)
        stderr = ""
        returncode = 0

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: MockProcessResult())

    # Call with default tolerance (DURATION_TOLERANCE = 2.0)
    report = verify_rendered_video(
        dummy_video,
        expected_audio_duration=audio_dur,
        expected_width=1920,
        expected_height=1080,
        expected_fps=30.0,
        expected_video_codec="libx264",
        expected_audio_codec="aac",
    )

    diff = round(abs(video_dur - audio_dur), 3)
    assert pytest.approx(report.duration_difference, abs=1e-3) == expected_diff
    assert pytest.approx(diff, abs=1e-3) == expected_diff

    if expected_pass:
        assert report.video_valid is True, f"Expected PASS for diff {diff}s <= 2.0s, but got errors: {report.errors}"
        assert len(report.errors) == 0
    else:
        assert report.video_valid is False, f"Expected FAIL for diff {diff}s > 2.0s, but it passed"
        assert len(report.errors) > 0
        assert any("tolerance: 2.000s" in err for err in report.errors), f"Error missing 'tolerance: 2.000s': {report.errors}"
        assert any(
            f"differs from audio duration ({audio_dur:.3f}s) by {expected_diff:.3f}s" in err
            for err in report.errors
        ), f"Error missing diff text: {report.errors}"


