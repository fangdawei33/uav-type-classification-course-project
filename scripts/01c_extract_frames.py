"""01c. Sample full RGB frames, retain source-video groups and create bbox previews."""
from _bootstrap import run_stage
from uav_project.video_frames import extract_frames

if __name__ == "__main__":
    run_stage("extract_frames", extract_frames)
