"""03f. Prepare or apply a fixed batch of externally acquired target boxes.

Preparation is not review. Apply takes explicit inspected normalized boxes,
preserves old records and refuses to label uncertain drone subtypes.
"""
import argparse
import json
from pathlib import Path
from _bootstrap import PROJECT_ROOT
from uav_project.common import META, OUT, rows, write_rows, digest, relative, save_json, now, resolve
from PIL import Image, ImageOps, ImageDraw, ImageFont

FOLDER=OUT/'review_batches/external_remaining_boxes_v1'

def prepare():
    reviews={r['sample_id']:r for r in rows(META/'drone_type_external_review.csv')}
    boxes={r['sample_id']:r for r in rows(META/'drone_type_external_box_proposals.csv')}
    raw=[r for name in ('drone_type_external_raw.csv','drone_type_article_raw.csv') for r in rows(META/name)]
    left=[r for r in raw if reviews.get(r['sample_id'],{}).get('visual_decision')=='candidate_keep' and boxes.get(r['sample_id'],{}).get('status')!='reviewed_geometry_only']
    FOLDER.mkdir(parents=True,exist_ok=True)
    manifest=FOLDER/'manifest.csv'
    if manifest.exists():
        raise FileExistsError('Existing review batch retained: '+str(manifest))
    records=[]
    for i,r in enumerate(left,1):
        records.append(dict(number=i,**r))
    write_rows(manifest,records)
    for start in range(0,len(records),8):
        sheet=Image.new('RGB',(1600,1760),'white')
        draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(records[start:start+8]):
            x=(j%2)*800;y=(j//2)*440
            with Image.open(resolve(r['path'])) as source:
                im=ImageOps.exif_transpose(source).convert('RGB')
                if digest(resolve(r['path']))!=r['sha256']:raise ValueError('Parent changed')
                thumb=ImageOps.contain(im,(760,350))
            ox=x+20+(760-thumb.width)//2;oy=y+30+(350-thumb.height)//2
            sheet.paste(thumb,(ox,oy))
            draw.rectangle((ox,oy,ox+thumb.width-1,oy+thumb.height-1),outline='#888888')
            for t in range(1,10):
                xx=ox+round(thumb.width*t/10);yy=oy+round(thumb.height*t/10)
                draw.line((xx,oy,xx,oy+5),fill='red');draw.line((ox,yy,ox+5,yy),fill='red')
            draw.text((x+10,y+8),str(r['number'])+' | '+r['candidate_class_name']+' | '+r['sample_id'],fill='black')
            draw.text((x+10,y+385),r['title'][:105],fill='black')
            draw.text((x+10,y+408),'red ticks: 10% of original image; '+str(im.size),fill='black')
        sheet.save(FOLDER/f'originals_{start//8+1:02}.jpg',quality=96)
    print('Prepared',len(records),'originals in',FOLDER)


def apply(annotation_path):
    annotations=json.loads(Path(annotation_path).read_text(encoding='utf-8'))
    manifest=rows(FOLDER/'manifest.csv');index={int(r['number']):r for r in manifest}
    old=rows(META/'drone_type_external_box_proposals.csv')
    existing={r['sample_id'] for r in old}
    if {int(a['number']) for a in annotations}!=set(index):raise ValueError('Explicit geometry required for entire fixed batch')
    backup=FOLDER/'before_boxes.csv'
    if backup.exists():raise FileExistsError('Batch already applied')
    new=[]
    for a in annotations:
        r=index[int(a['number'])]
        if r['sample_id'] in existing:raise ValueError('Refuse overwrite of previous box')
        with Image.open(resolve(r['path'])) as source:
            im=ImageOps.exif_transpose(source).convert('RGB')
        if digest(resolve(r['path']))!=r['sha256']:raise ValueError('Original changed')
        box=tuple(round(float(v)*(im.width if k%2==0 else im.height)) for k,v in enumerate(a['xyxy']))
        x1,y1,x2,y2=box
        if not (0<=x1<x2<=im.width and 0<=y1<y2<=im.height):raise ValueError('Bounds out of image')
        marked=im.copy();ImageDraw.Draw(marked).rectangle((x1,y1,x2-1,y2-1),outline='red',width=max(2,im.width//300))
        card=Image.new('RGB',(1200,460),'white')
        for offset,part in [(0,marked),(600,im.crop(box))]:
            thumb=ImageOps.contain(part,(590,400));card.paste(thumb,(offset+(600-thumb.width)//2,28+(400-thumb.height)//2))
        draw=ImageDraw.Draw(card);draw.text((5,5),str(a['number'])+' | '+r['sample_id'],fill='black');draw.text((5,435),str(box)+'; geometry only; original labels unchanged',fill='black')
        path=FOLDER/(r['sample_id']+'_box.jpg');card.save(path,quality=95)
        new.append({'sample_id':r['sample_id'],'class_id':r['candidate_label'],'x_min':x1,'y_min':y1,'x_max_exclusive':x2,'y_max_exclusive':y2,'coordinate_space':'EXIF-transposed RGB display pixels','image_sha256':r['sha256'],'status':'reviewed_geometry_only','reviewer':'Codex_AI_assisted_visual_review','evidence':relative(path),'evidence_sha256':digest(path)})
    write_rows(backup,old);write_rows(META/'drone_type_external_box_proposals.csv',old+new)
    write_rows(FOLDER/'new_boxes.csv',new)
    # These cards remain proposals until the crop check step has actually inspected them.
    for start in range(0,len(new),4):
        sheet=Image.new('RGB',(1200,1840),'white')
        for j,r in enumerate(new[start:start+4]):
            with Image.open(resolve(r['evidence'])) as card:sheet.paste(card,(0,j*460))
        sheet.save(FOLDER/f'crops_{start//4+1:02}.jpg',quality=95)
    save_json(FOLDER/'geometry_record.json',{'created_at':now(),'images':len(new),'annotation_sha256':digest(Path(annotation_path)),'original_contact_sheets_inspected':True,'crop_contact_sheets_inspected':False,'AI_assisted_not_expert_authentication':True,'label_or_split_changed':False})
    print('Recorded',len(new),'boxes; inspect generated crop sheets before release.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--annotations');a=p.parse_args()
    if a.annotations:apply(a.annotations)
    else:prepare()
