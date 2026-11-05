"""Development-set error-analysis diagnostics ONLY; never generates or fits weights.

The assistant used its printed summaries to decide manually written features.
This is not called at inference time or during official test evaluation.
"""
import numpy as np,cv2
from scipy.ndimage import label
from pathlib import Path
with np.load(Path(__file__).resolve().parents[1]/'data/mnist.npz') as z:
 rng=np.random.default_rng(20261105)
 ii=rng.permutation(60000)[:2000]
 images=z['x_train'][ii]; y=z['y_train'][ii]

def features(a):
 im=(a>75).astype(np.uint8)
 yy,xx=np.where(im)
 if not len(xx):return [0]*8
 x0,x1=xx.min(),xx.max();y0,y1=yy.min(),yy.max();cx=(x0+x1)/2;cy=(y0+y1)/2
 top=im[y0:y0+6,:];btm=im[max(y0,y1-5):y1+1,:]
 topbar=max([row.sum() for row in top],default=0)
 bottombar=max([row.sum() for row in btm],default=0)
 upper=im[y0:int(cy)+1]
 left=int(upper[:,:int(cx)].sum());right=int(upper[:,int(cx):].sum())
 upperbalance=(left-right)/max(1.,left+right)
 lower=im[int(cy):y1+1]
 lowerwide=int((lower[:,:int(cx)].sum()+lower[:,int(cx):].sum()) / max(1,lower.size))
 # holes as connectedComponents of inverse binary
 components,labs,stats,centroids=cv2.connectedComponentsWithStats(1-im,connectivity=8)
 holes=[(stats[i,4],centroids[i,1]) for i in range(1,components) if stats[i,0]>0 and stats[i,1]>0 and stats[i,0]+stats[i,2]<28 and stats[i,1]+stats[i,3]<28]
 holes.sort(reverse=True)
 holey=(holes[0][1]-y0)/max(1.,y1-y0) if holes else -1
 # pixels in bottom-left; bottom-right
 lowleft=float(lower[:,:int(cx)].sum())/max(1,lower.sum())
 upperleft=float(upper[:,:int(cx)].sum())/max(1,upper.sum())
 return [topbar,bottombar,upperbalance,lowleft,upperleft,holey,float(x1-x0),float(y1-y0)]

F=np.array([features(a) for a in images]);names='topbar bottombar upper_balance lower_left_fraction upper_left_fraction hole_y_relative width height'.split()
for j,n in enumerate(names):
 print('\n',n)
 print('class | p25 / median / p75')
 for i in range(10):
  v=F[y==i,j]; print(i,*[round(float(x),2) for x in np.quantile(v,[.25,.5,.75])])
