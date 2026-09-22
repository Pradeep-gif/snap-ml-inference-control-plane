# ML Inference Control Plane

A dependency-free Python prototype for latency-sensitive model serving and vector retrieval.

## Features

- Weighted model-version routing.
- SLO-aware request admission.
- Bounded dynamic batching.
- Cosine-similarity vector retrieval and Recall@K.
- Counters for accepted, rejected, and completed requests.

## Run tests

python -m unittest discover -s tests -v

The scheduler uses predicted queue delay and batch runtime to protect a latency SLO. Production systems would source estimates from live histograms, dispatch to GPU workers, and validate quality and latency during rollout.
