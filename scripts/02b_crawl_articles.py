"""02b. Discover configured original article media, then download explicitly selected candidates."""
import argparse
from _bootstrap import run_stage
from uav_project.article_crawler import collect_articles

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-id', help='Only one source ID in configs/article_sources.json')
    parser.add_argument('--download-approved', action='store_true',
                        help='Download inventory rows explicitly selected for collection; labels remain unverified')
    parser.add_argument('--use-saved-page', action='store_true',
                        help='Parse only the configured, hash-verified HTML snapshot; makes no network requests')
    parser.add_argument('--group', help='Only process sources in this explicitly configured collection group')
    args = parser.parse_args()
    if args.use_saved_page and (not args.source_id or args.download_approved):
        parser.error('--use-saved-page requires --source-id and cannot download media')
    run_stage('crawl_external', lambda: collect_articles(args.download_approved, args.source_id, args.use_saved_page, args.group))
