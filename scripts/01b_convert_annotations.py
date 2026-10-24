"""01b. Convert original MAT annotations; RGB output requires visual verification."""
from _bootstrap import run_stage
from uav_project.video_annotations import convert_annotations

if __name__ == "__main__":
    run_stage("convert_annotations", convert_annotations)
