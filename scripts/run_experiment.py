import argparse,sys,os,json,hashlib,datetime
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.model import compile_weights,predict
p=argparse.ArgumentParser()
p.add_argument('--name',required=True);p.add_argument('--mode',choices=['dev','test'],default='dev');p.add_argument('--n',type=int,default=2000);p.add_argument('--gpu',action='store_true')
a=p.parse_args()
repo=Path(__file__).resolve().parents[1]
with np.load(repo/'data/mnist.npz') as z:
 if a.mode=='test':xx,yy=z['x_test'],z['y_test']
 else:
  rng=np.random.default_rng(20261105)
  indices=rng.permutation(60000)[:5000]
  xx,yy=z['x_train'][indices],z['y_train'][indices]
xx=xx[:a.n]; yy=yy[:a.n]
weights=compile_weights()
scores,closest=predict(xx,weights,device='gpu' if a.gpu else 'cpu')
pred=scores.argmin(1)
correct=(pred==yy)
out=repo/'experiments'/a.name
out.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out/'weights.npz',**weights)
np.savez_compressed(out/'evaluation.npz',actual=yy,predicted=pred,score=scores,prototype_index=closest)
from src.manual_shapes import STROKES,GEOMETRY,COEFFICIENTS,HOLE_PENALTIES,ENDPOINT_TARGETS,ENDPOINT_WEIGHTS,ROW_TARGETS,ROW_WEIGHTS,GEOMETRIC_TARGETS,GEOMETRIC_WEIGHTS
meta=dict(name=a.name,mode=a.mode,samples=len(yy),accuracy=float(correct.mean()),errors=int((~correct).sum()),dataset_sha256=hashlib.sha256((repo/'data/mnist.npz').read_bytes()).hexdigest(),prototypes=int(len(weights['labels'])),weights_shapes={k:list(v.shape) for k,v in weights.items()},gpu=a.gpu,coefficients=COEFFICIENTS,hole_penalties=HOLE_PENALTIES,endpoint_targets=ENDPOINT_TARGETS,endpoint_weights=ENDPOINT_WEIGHTS,row_targets=ROW_TARGETS,row_weights=ROW_WEIGHTS,geometric_targets=GEOMETRIC_TARGETS,geometric_weights=GEOMETRIC_WEIGHTS,geometry=GEOMETRY,n_styles={str(k):len(v) for k,v in STROKES.items()},timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),per_class={str(i):dict(n=int((yy==i).sum()),accuracy=float(correct[yy==i].mean())) for i in range(10)})
(out/'metrics.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
with open(out/'misclassified.txt','w') as f:
 for i in np.flatnonzero(~correct)[:90]:
  f.write(f'#{i} actual={yy[i]} guessed={pred[i]} best_score={scores[i,pred[i]]:.3f} alt={np.argsort(scores[i])[:3].tolist()}\n')
  for row in xx[i]:f.write(''.join('##' if p>120 else '++' if p>50 else '  ' for p in row)+'\n')
  f.write('\n')
print(json.dumps({k:meta[k] for k in ['name','mode','samples','accuracy','errors','prototypes','per_class']},indent=2))
