[![Paper](https://img.shields.io/badge/paper-arXiv%3A1912.12002-B31B1B.svg)](https://arxiv.org/abs/1912.12002)

# Quantum Logic Gate Synthesis as a Markov Decision Process

M. Sohaib Alam, [Noah F. Berthusen](https://noahberthusen.github.io), [Peter P. Orth](https://faculty.sites.iastate.edu/porth/)

### Abstract 
Reinforcement learning has witnessed recent applications to a variety of tasks in quantum programming. The underlying assumption is that those tasks could be modeled as Markov Decision Processes (MDPs). Here, we investigate the feasibility of this assumption by exploring its consequences for two fundamental tasks in quantum programming: state preparation and gate compilation. By forming discrete MDPs, focusing exclusively on the single-qubit case (both with and without noise), we solve for the optimal policy exactly through policy iteration. We find optimal paths that correspond to the shortest possible sequence of gates to prepare a state, or compile a gate, up to some target accuracy. As an example, we find sequences of $H$ and $T$ gates with length as small as $11$ producing $\sim 99\%$ fidelity for states of the form $(HT)^{n} |0\rangle$ with values as large as $n=10^{10}$. In the presence of gate noise, we demonstrate how the optimal policy adapts to the effects of noisy gates in order to achieve a higher state fidelity. This work provides strong evidence that reinforcement learning can be used for optimal state preparation and gate compilation for larger qubit spaces.

### Description
This repository includes information, code, scripts, and data to generate the figures in the paper.

### Requirements
