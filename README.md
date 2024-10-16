# EEG microstate transition cost & task demands

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13709700.svg)](https://doi.org/10.5281/zenodo.13709700)

Solving the Schr&ouml;dinger bridge problem for computing brain transition cost from neurophysiological data.

See our work: Barzon, G., Ambrosini, E., Vallesi, A., & Suweis, S. (2024). [EEG microstate transition cost correlates with task demands](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012521). PLOS Comp. Bio., 2024.

![alt text](/../master/images/figure1.png?raw=true)

## Installation

The library relies on the following Python packages:
- [MNE](https://mne.tools/stable/index.html): package for exploring, visualizing, and analyzing human neurophysiological data
- [Pycrostates](https://pycrostates.readthedocs.io/en/latest/): package for analyzing EEG microstates
- [POT](https://github.com/PythonOT/POT): Python Optimal Transport

## Examples

* Compute states of brain activity w clustering algorithms and related statistics: [Pycrostates_clustering.ipynb](/../master/Pycrostates_clustering.ipynb)
* Compute brain control cost in different conditions: [Pycrostates_data_analysis.ipynb](/../master/Pycrostates_data_analysis.ipynb)
* Visualize the results: [Figures.ipynb](/../master/Figures.ipynb)

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

[4] Férat, V., Scheltienne, M., Brunet, D., Ros, T., & Michel, C. (2022). [Pycrostates: a Python library to study EEG microstates](https://joss.theoj.org/papers/10.21105/joss.04564). Journal of Open Source Software, 7(78), 4564.
