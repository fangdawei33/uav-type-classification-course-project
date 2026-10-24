"""02c. Discover links on explicitly configured public pages; do not collect images."""
import argparse
from _bootstrap import run_stage
from uav_project.source_discovery import discover_source_links

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-id')
    parser.add_argument('--group', help='Only discover the named configured source group')
    args=parser.parse_args()
    run_stage('crawl_external',lambda:discover_source_links(args.source_id,args.group))
