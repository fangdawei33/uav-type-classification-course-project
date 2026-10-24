"""04h. Materialize native target crops and image-quality metadata; no models."""
from _bootstrap import PROJECT_ROOT
from uav_project.expanded_processing import process_images

if __name__ == '__main__':
    process_images()
