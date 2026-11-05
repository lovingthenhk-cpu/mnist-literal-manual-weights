# MNIST: explicit hand-authored numerical weights, no learning or reference-image retrieval

**This is a completely separate, non-continuation experiment.** `/mnist-direct-weight-edit-experiments` and its 99.02% output are INVALID for this task. We neither import weights nor examples from that repository.

- 10-way classification of official MNIST 28×28 test images.
- Every prototype is specified as *literal, human-authored numeric line coordinates* in `src/manual_shapes.py`. Those coordinates are deterministically rasterized into explicit 784-entry weights per prototype and 784-entry asymmetric-distance weights; biases and combination coefficients are manually typed there too.
- No gradient descent, backprop, SGD, Adam, genetic algorithm, coordinate-descent search, training-data-derived prototypes/centroids, kNN using MNIST samples, pretrained models, or other parameter fitting. MNIST training data is **only** inspected as a development evaluation set.
- A fixed, separately withheld 10,000-item test split is evaluated once after manual decisions are frozen.
- Each experiment is committed via its own GitHub pull request, including its complete compressed numerical weight arrays, metrics, and logged misclassifications. Bad experiments are retained.
- Goal: **≥90.00% accuracy**; any lower accuracy is reported without embellishment.

## Run

```sh
python3 -m pip install numpy scipy scikit-image opencv-python-headless cupy-cuda13x
python3 - <<'PY'
import urllib.request
urllib.request.urlretrieve('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz','data/mnist.npz')
PY
python3 scripts/run_experiment.py --name exp001 --gpu --n 2000
```

`weights.npz` contains real, reconstructable arrays; these are not pre-trained weights. All shape-control values and coefficients are readable in source.
