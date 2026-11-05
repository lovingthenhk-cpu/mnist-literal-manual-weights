"""Deterministic, train-free handwritten-stroke digit classifier.
Only parameters in manual_shapes.py may change; NO learner or optimizer.
"""
import numpy as np, cv2
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from .manual_shapes import STROKES, GEOMETRY, COEFFICIENTS

N=28

def catmull(pts, steps=7):
    pts=np.array(pts, dtype=np.float32)
    if len(pts)<3:return pts
    p=np.vstack([pts[0],pts,pts[-1]])
    parts=[]
    for j in range(1,len(p)-2):
        a,b,c,d=p[j-1:j+3]
        for t in np.linspace(0,1,steps,endpoint=False):
            parts.append(0.5*((2*b)+(-a+c)*t+(2*a-5*b+4*c-d)*t*t+(-a+3*b-3*c+d)*t*t*t))
    parts.append(pts[-1]);return np.array(parts)

def make_templates():
    shapes=[]; names=[]; labels=[]; penalties=[]
    for digit,styles in STROKES.items():
        for k,subpaths in enumerate(styles):
            for sx in GEOMETRY['width']:
                for tilt in GEOMETRY['tilt']:
                    for sy in GEOMETRY['height']:
                        for dy in GEOMETRY['dy']:
                            for dx in GEOMETRY['dx']:
                                img=np.zeros((N,N),np.uint8)
                                for smooth,points in subpaths:
                                    pts=catmull(points) if smooth else np.array(points,dtype=np.float32)
                                    x=14+(pts[:,0]-.5)*19*sx+tilt*(pts[:,1]-.5)*20+dx
                                    y=14+(pts[:,1]-.5)*23*sy+dy
                                    q=np.rint(np.stack([x,y],-1)*8).astype(np.int32)
                                    cv2.polylines(img,[q],False,255,thickness=2,lineType=cv2.LINE_AA,shift=3)
                                center=skeletonize(img>90).astype(np.float32)
                                if center.sum()<5:continue
                                labels.append(digit);shapes.append(center)
                                names.append(f'{digit}:style{k}:w{sx}:sl{tilt}:h{sy}:dy{dy}:dx{dx}')
                                penalties.append(abs(sx-1)+abs(tilt)+abs(sy-1)+abs(dy)/15+abs(dx)/15)
    return np.array(shapes,dtype=np.float32),np.array(labels),names,np.array(penalties,np.float32)

def hole_count(im):
    bin=(im>65).astype(np.uint8)
    # Close narrow open loops. Hand-coded topological feature, no fit.
    bin=cv2.morphologyEx(bin,cv2.MORPH_CLOSE,np.ones((2,2),np.uint8))
    comps,lab,stats,centroids=cv2.connectedComponentsWithStats(1-bin,connectivity=8)
    return sum(int(not ((stats[i,cv2.CC_STAT_LEFT]==0) or (stats[i,cv2.CC_STAT_TOP]==0) or
            (stats[i,cv2.CC_STAT_LEFT]+stats[i,cv2.CC_STAT_WIDTH]>=28) or
            (stats[i,cv2.CC_STAT_TOP]+stats[i,cv2.CC_STAT_HEIGHT]>=28))) for i in range(1,comps))

def preprocess(X):
    # No labels are needed. Preprocess each example independently.
    sk=[];dist=[];holes=[]
    for a in X:
        bin=(a>65)
        skel=skeletonize(bin)
        if skel.sum()==0:skel=bin
        sk.append(skel.astype(np.float32))
        dist.append(distance_transform_edt(~bin).astype(np.float32))
        holes.append(hole_count(a))
    return np.array(sk),np.array(dist),np.array(holes,np.int32)

def compile_weights():
    templates,labels,names,penalties=make_templates()
    dists=np.array([distance_transform_edt(1-a) for a in templates],dtype=np.float32)
    wforward=templates.reshape(len(templates),-1)
    wforward=wforward/np.maximum(wforward.sum(axis=1,keepdims=True),1)
    wreverse=dists.reshape(len(templates),-1)
    return dict(forward_weights=wforward,reverse_weights=wreverse,
                labels=labels.astype(np.int8),names=np.array(names),
                geom_penalty=penalties)

def predict(X,weights=None,device='auto'):
    w=weights if weights is not None else compile_weights()
    images=np.asarray(X)
    sk,dt,holes=preprocess(images)
    backend='numpy'
    if device!='cpu':
        try:
            import cupy as xp
            if xp.cuda.runtime.getDeviceCount()>0:backend='cupy'
        except Exception:
            if device=='gpu':raise
    if backend=='numpy':import numpy as xp
    else: import cupy as xp
    T=xp.asarray(w['forward_weights'].T)
    R=xp.asarray(w['reverse_weights'].T)
    labels=np.asarray(w['labels'])
    g=xp.asarray(w['geom_penalty'])
    scores_all=[]; argall=[]
    biases=COEFFICIENTS['class_bias']
    for start in range(0,len(images),512):
        end=min(start+512,len(images))
        sks=xp.asarray(sk[start:end].reshape(-1,784),dtype=xp.float32)
        dts=xp.asarray(dt[start:end].reshape(-1,784),dtype=xp.float32)
        a=(dts@T) * COEFFICIENTS['template_to_input']
        b=(sks@R)/xp.maximum(sks.sum(axis=1,keepdims=True),1) * COEFFICIENTS['input_to_template']
        raw=a+b+g[None,:]*COEFFICIENTS['geometric_penalty']
        h=xp.asarray(holes[start:end])
        all_classes=[]; all_arg=[]
        for cls in range(10):
            indices=np.flatnonzero(labels==cls)
            cost=raw[:,indices]
            m=xp.min(cost,axis=1)
            arg=xp.argmin(cost,axis=1)
            # For class 4 the holes feature has ambiguity; explicitly prescribed penalty.
            allowed={0:(1,),1:(0,),2:(0,1),3:(0,),4:(0,1),5:(0,),6:(0,1),7:(0,),8:(1,2),9:(0,1,)}
            penalty=xp.min(xp.stack([xp.abs(h-a) for a in allowed[cls]],axis=1),axis=1)*COEFFICIENTS['holes']
            all_classes.append(m+penalty+biases[cls]);all_arg.append(xp.asarray(indices)[arg])
        scores_all.append(xp.asnumpy(xp.stack(all_classes,axis=1)) if backend=='cupy' else xp.stack(all_classes,axis=1))
        argall.append(xp.asnumpy(xp.stack(all_arg,axis=1)) if backend=='cupy' else xp.stack(all_arg,axis=1))
    return np.concatenate(scores_all),np.concatenate(argall)
