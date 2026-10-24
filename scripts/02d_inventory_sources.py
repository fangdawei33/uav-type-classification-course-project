"""02d. Count eligible file links in configured Commons categories; no media download."""
import argparse
from _bootstrap import run_stage
from uav_project.source_inventory import inventory_commons

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--class-id',type=int,choices=range(7))
    parser.add_argument('--max-pages-per-seed',type=int,default=10)
    args=parser.parse_args()
    if args.max_pages_per_seed<1:parser.error('Page limit must be positive')
    run_stage('crawl_external',lambda:inventory_commons(args.class_id,args.max_pages_per_seed))
