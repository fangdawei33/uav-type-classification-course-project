"""HOG, HSV, LBP and image-quality features; no title/source metadata in X."""
import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog, local_binary_pattern
from .common import OUT, resolve, save_json
from .splitting import load_manifest

def features(frame):
    cache = OUT / "feature_cache"; cache.mkdir(exist_ok=True)
    vectors = []
    for _,r in frame.iterrows():
        p = cache / ("hog_v2_" + r.clean_sha256 + ".npz")
        if p.exists():
            a = np.load(p, allow_pickle=False); vectors.append(a["x"]); names = a["names"].tolist(); continue
        with Image.open(resolve(r.clean_path)) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            ratio = im.width/im.height
            rgb = np.asarray(im.resize((128,128)), dtype=float)/255.
        hsv = rgb2hsv(rgb); gray = rgb2gray(rgb); x=[]; names=[]
        for i,channel in enumerate(["hue","saturation","value"]):
            x += [float(hsv[:,:,i].mean()), float(hsv[:,:,i].std())]
            names += [channel+"_mean", channel+"_std"]
            hist = np.histogram(hsv[:,:,i], bins=16, range=(0,1))[0].astype(float)
            x.extend(hist/max(hist.sum(),1)); names.extend([f"{channel}_bin_{j}" for j in range(16)])
        h = hog(rgb, orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2), channel_axis=2)
        x.extend(h); names.extend([f"hog_{j}" for j in range(len(h))])
        lbp = local_binary_pattern((gray*255).astype(np.uint8),8,1,"uniform")
        hist=np.histogram(lbp,bins=10,range=(0,10))[0].astype(float)
        x.extend(hist/max(hist.sum(),1)); names.extend([f"lbp_{j}" for j in range(10)])
        x.extend([gray.mean(),gray.std(),ratio]); names.extend(["brightness","contrast","aspect_ratio"])
        x=np.asarray(x,dtype=np.float32)
        np.savez_compressed(p,x=x,names=np.asarray(names)); vectors.append(x)
    return np.vstack(vectors), names



def extract_training_features():
    frame = load_manifest()
    # Test features are only computed during final evaluation.
    selected = frame[frame.split.isin(["train", "validation"])]
    matrix, names = features(selected)
    save_json(OUT / "feature_schema.json", {"names": names, "samples": len(selected), "dimensions": matrix.shape[1]})
    print("开发集特征已缓存：", matrix.shape, OUT / "feature_cache")
