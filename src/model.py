"""Deterministic, train-free handwritten-stroke digit classifier.
Only parameters in manual_shapes.py may change; NO learner or optimizer.
"""
import numpy as np, cv2
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from .manual_shapes import STROKES, GEOMETRY, COEFFICIENTS, HOLE_PENALTIES

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
    sk=[];dist=[];holes=[];geometry=[]
    for a in X:
        bin=(a>65)
        skel=skeletonize(bin)
        if skel.sum()==0:skel=bin
        sk.append(skel.astype(np.float32))
        dist.append(distance_transform_edt(~bin).astype(np.float32))
        holes.append(hole_count(a))
        ys,xs=np.where(bin)
        if len(ys):
            x0,x1=xs.min(),xs.max();y0,y1=ys.min(),ys.max()
            cx=(x0+x1)/2;cy=(y0+y1)/2
            top=bin[y0:min(y0+6,28)]
            bottom=bin[max(y0,y1-5):y1+1]
            topbar=float(max((row.sum() for row in top),default=0))
            bottombar=float(max((row.sum() for row in bottom),default=0))
            upper=bin[y0:int(cy)+1]
            ul=float(upper[:,:int(cx)].sum())/max(1.,float(upper.sum()))
            lower=bin[int(cy):y1+1]
            ll=float(lower[:,:int(cx)].sum())/max(1.,float(lower.sum()))
            comps,_,stats,centroids=cv2.connectedComponentsWithStats(1-bin.astype(np.uint8),connectivity=8)
            hlist=[(stats[i,4],centroids[i,1]) for i in range(1,comps)
                if stats[i,0]>0 and stats[i,1]>0 and stats[i,0]+stats[i,2]<28 and stats[i,1]+stats[i,3]<28]
            largest=max(hlist,key=lambda t:t[0]) if hlist else None
            holey=float((largest[1]-y0)/max(1,y1-y0)) if largest else -1.
            mid=bin[int(y0+.43*(y1-y0)):min(28,int(y0+.72*(y1-y0))+1)]
            midbar=float(max((row.sum() for row in mid),default=0))
            geometry.append([topbar,bottombar,ul,ll,holey,float(x1-x0),midbar])
        else:geometry.append([0.,0.,0.,0.,-1.,0.,0.])
    return np.array(sk),np.array(dist),np.array(holes,np.int32),np.array(geometry,np.float32)

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
    sk,dt,holes,geometry=preprocess(images)
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
        geom=xp.asarray(geometry[start:end]);t,b,ul,ll,hy,ww,mid=[geom[:,i] for i in range(7)]
        all_classes=[]; all_arg=[]
        for cls in range(10):
            indices=np.flatnonzero(labels==cls)
            cost=raw[:,indices]
            m=xp.min(cost,axis=1)
            arg=xp.argmin(cost,axis=1)
            # For class 4 the holes feature has ambiguity; explicitly prescribed penalty.
            penalty=xp.asarray(HOLE_PENALTIES[cls])[xp.minimum(h,3)]
            # Hand-authored spatial discrimination; all weights are literal constants.
            # 0 vs 6/9: loop vertical position; 1 vs 7: top bar.
            if cls==0:
                penalty+=.14*xp.maximum(8-t,0)+.11*xp.maximum(9-b,0)
                penalty+=xp.where((hy>=0)&((hy>.64)|(hy<.40)),.70,0.)
            elif cls==1:
                penalty+=.26*xp.maximum(t-7,0.)+.20*xp.maximum(b-7,0.)+2.0*xp.maximum(ul-.27,0.)
            elif cls==2:
                penalty+=.22*xp.maximum(12-b,0.)+.15*xp.maximum(8-t,0.)+5.0*xp.maximum(.43-ll,0.)
            elif cls==3:
                penalty+=5.0*xp.maximum(ul-.43,0.)+.10*xp.maximum(b-13,0.)+.20*xp.maximum(7-t,0.)+4.0*xp.maximum(ll-.43,0.)
            elif cls==4:
                penalty+=.17*xp.maximum(b-7,0.)+.15*xp.maximum(t-9,0.)+.20*xp.maximum(5-t,0.)+3.0*xp.maximum(.28-ul,0.)+.12*xp.maximum(8-mid,0.)+xp.where((hy>=0)&(hy<.38),.60,0.)
            elif cls==5:
                penalty+=4.5*xp.maximum(.40-ul,0.)+2.0*xp.maximum(.36-ll,0.)+.18*xp.maximum(10-t,0.)
            elif cls==6:
                penalty+=.14*xp.maximum(t-8,0.)+.14*xp.maximum(8-b,0.)
                penalty+=xp.where((hy>=0)&(hy<.60),.55,0.)
            elif cls==7:
                penalty+=.22*xp.maximum(11-t,0.)+.16*xp.maximum(b-8,0.)
            elif cls==8:
                penalty+=.18*xp.maximum(7-t,0.)+.42*xp.maximum(b-11,0.)+.18*xp.maximum(ww-18,0.)
            elif cls==9:
                penalty+=.18*xp.maximum(b-6,0.)
                penalty+=xp.where((hy>=0)&(hy>.43),.65,0.)
            all_classes.append(m+penalty+biases[cls]);all_arg.append(xp.asarray(indices)[arg])
        scores_all.append(xp.asnumpy(xp.stack(all_classes,axis=1)) if backend=='cupy' else xp.stack(all_classes,axis=1))
        argall.append(xp.asnumpy(xp.stack(all_arg,axis=1)) if backend=='cupy' else xp.stack(all_arg,axis=1))
    return np.concatenate(scores_all),np.concatenate(argall)
