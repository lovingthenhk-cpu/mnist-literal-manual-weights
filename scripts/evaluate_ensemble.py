"""Evaluation only: fixed 50/50 average of two explicitly authored models.

Never learns a weight. It does not read previously computed prediction outputs.
It loads each compiled numerical matrix from exp016/exp017, reruns inference,
and uses only literally-written ensemble coefficients [0.5,0.5].
"""
import argparse,hashlib,datetime,json,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from src.model import predict

p=argparse.ArgumentParser()
p.add_argument('--name',required=True)
p.add_argument('--mode',choices=['dev','test'],default='dev')
p.add_argument('--n',type=int,default=2000)
p.add_argument('--gpu',action='store_true')
a=p.parse_args()
with np.load(REPO/'data/mnist.npz') as z:
    if a.mode=='test':X,y=z['x_test'][:a.n],z['y_test'][:a.n]
    else:
        order=np.random.default_rng(20261105).permutation(60000)[:5000]
        X,y=z['x_train'][order][:a.n],z['y_train'][order][:a.n]
weights=[]
for name in ['exp016','exp017']:
    with np.load(REPO/'experiments'/name/'weights.npz') as z:
        weights.append({k:z[k].copy() for k in z.files})
# Literal, unoptimized coefficients. Equal blend; never estimated on examples.
blend=np.array([.5,.5],np.float32)
sc=[]
for w in weights:
    score,_=predict(X,weights=w,device='gpu' if a.gpu else 'cpu')
    sc.append(score)
score=blend[0]*sc[0]+blend[1]*sc[1]
pred=score.argmin(axis=1)
acc=float((pred==y).mean())
out=REPO/'experiments'/a.name
out.mkdir(parents=True,exist_ok=True)
merged={'blend':blend}
for j,w in enumerate(weights):
    for k,v in w.items():merged[f'model{j}_{k}']=v
np.savez_compressed(out/'weights.npz',**merged)
np.savez_compressed(out/'evaluation.npz',actual=y,predicted=pred,score=score,
                    score_model016=sc[0],score_model017=sc[1])
metrics=dict(name=a.name,mode=a.mode,n=int(len(y)),correct=int((pred==y).sum()),
    wrong=int((pred!=y).sum()),accuracy=acc,
    model_sources=['exp016','exp017'],ensemble_coefficients=blend.tolist(),
    data_sha256=hashlib.sha256((REPO/'data/mnist.npz').read_bytes()).hexdigest(),
    gpu=bool(a.gpu), timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    classes={str(i):{'count':int((y==i).sum()),'correct':int(((y==i)&(pred==i)).sum()),
    'accuracy':float((pred[y==i]==i).mean())} for i in range(10)})
(out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
with (out/'misclassified.txt').open('w') as f:
    for i in np.flatnonzero(pred!=y)[:100]:
        f.write(f'#{i}: actual={y[i]} predicted={pred[i]} top3={np.argsort(score[i])[:3].tolist()}\n')
        for row in X[i]:f.write(''.join('##' if v>120 else '++' if v>50 else '  ' for v in row)+'\n')
        f.write('\n')
print(json.dumps(metrics,indent=2))
