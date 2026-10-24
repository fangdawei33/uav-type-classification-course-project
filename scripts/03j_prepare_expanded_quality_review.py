"""03j. Prepare low-contrast/dark crop review pages after 04h; no classification."""
from _bootstrap import PROJECT_ROOT
from uav_project.common import rows, resolve, read_json, digest
from uav_project.expanded_processing import settings
from uav_project.expanded_release import prepare_quality_queue
from PIL import Image, ImageDraw, ImageOps

if __name__ == '__main__':
    work = resolve(settings()['work_directory'])
    source = work / 'processed_images.csv'
    receipt = read_json(work / 'processing_receipt.json')
    if digest(source) != receipt['manifest_sha256']:
        raise ValueError('Processing manifest changed')
    records = rows(source)
    count = prepare_quality_queue(records, work)
    flagged = rows(work / 'quality_review_queue.csv')
    for start in range(0, len(flagged), 8):
        sheet = Image.new('RGB', (1600, 1640), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, r in enumerate(flagged[start:start+8]):
            y = i*200
            with Image.open(resolve(r['raw_path'])) as original:
                marked = original.convert('RGB')
                rect = tuple(map(int, r['crop_xyxy'].split(',')))
                ImageDraw.Draw(marked).rectangle(rect, outline='red', width=3)
                thumb = ImageOps.contain(marked, (780, 165))
                sheet.paste(thumb, ((800-thumb.width)//2, y))
            with Image.open(resolve(r['image_path'])) as crop:
                thumb = ImageOps.contain(crop, (780,165), method=Image.Resampling.NEAREST)
                sheet.paste(thumb, (800+(800-thumb.width)//2, y))
            draw.text((8,y+170), r['sample_id']+' | '+r['raw_path'].split('/')[-1], fill='black')
        sheet.save(work / f'quality_context_{start//8+1:02}.jpg', quality=95)
    print('Prepared quality-flagged queue:', count, '; pages are not yet reviewed.', flush=True)
