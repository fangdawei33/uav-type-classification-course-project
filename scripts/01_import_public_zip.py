"""01 alternative. Import the author's browser ZIP(s), reusing existing originals."""
import argparse
from pathlib import Path
from _bootstrap import run_stage
from uav_project.archive_import import import_public_zips

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--zip-dir", type=Path, help="只放作者原始数据ZIP的目录；可包含多份ZIP，不用解压")
    source.add_argument("--archives", type=Path, nargs="+", help="显式指定一份或多份ZIP")
    parser.add_argument("--accept-listed-subset", action="store_true", help="接受当前可访问子集，保留与论文数量的差异")
    args = parser.parse_args()
    archives = sorted(args.zip_dir.glob("*.zip")) if args.zip_dir else args.archives
    run_stage("import_public_zip", lambda: import_public_zips(archives, args.accept_listed_subset))
