# Brain control cost

Solving the Schr&ouml;dinger bridge problem for computing brain control cost.

![alt text](/../master/images/figure1.png?raw=true)

## Installation

The library relies on the following Python modules:
- Numpy
- Scipy
- Pandas
- MNE
- [POT](https://github.com/PythonOT/POT): Python Optimal Transport

## Examples

#### Short example

```python
# rest,task are 1D probability distributions (sum to 1 and positive)
# Qij is the co-occurrence matrix at resting (sum to 1 and positive)
cost_matrix = -np.log(Qij)
Pij = ot.optim.cg(rest, task, cost_matrix, 1, f, df, verbose=True)
cost = KL(Pij, Qij)
```

## References

[1] Chen, Y., Georgiou, T. T., & Pavon, M. (2016). [On the relation between optimal transport and Schrödinger bridges: A stochastic control viewpoint](https://link.springer.com/article/10.1007/s10957-015-0803-z). Journal of Optimization Theory and Applications, 169, 671-691.

[2] Cuturi, M. (2013). [Sinkhorn distances: Lightspeed computation of optimal transport](https://arxiv.org/pdf/1306.0895.pdf). In Advances in Neural Information Processing Systems (pp. 2292-2300).

[3] Kawakita, G., Kamiya, S., Sasai, S., Kitazono, J., & Oizumi, M. (2022). [Quantifying brain state transition cost via Schrödinger bridge](https://direct.mit.edu/netn/article/6/1/118/107814/Quantifying-brain-state-transition-cost-via). Network Neuroscience, 6(1), 118-134.
