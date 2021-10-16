This is the code for causal computatation of linear attention with the "sum" update rule.
We use this code for any model which :

* Linear Transformer
* Performer
* DPFP-based linear Transformer

The code in this directory is copied from [idiap/fast-transformers](https://github.com/idiap/fast-transformers/tree/master/fast_transformers/causal_product).

The only change we applied to this code is the support for carrying fast weight memory state across batches.
