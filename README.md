# L2HMC

Work-in-progress code for an implementation in `PyMC3` of the sampler from

[*Generalizing Hamiltonian Monte Carlo with Neural Network*](https://arxiv.org/abs/1711.09268)

by [Daniel Levy](http://ai.stanford.edu/~danilevy), [Matt D. Hoffman](http://matthewdhoffman.com/) and [Jascha Sohl-Dickstein](sohldickstein.com).

## Outline

In the paper, the authors propose generalizing the energy used in Hamiltonian Monte Carlo with six auxiliary functions to allow for transitions which, in particular, are not volume preserving. By restricting this generalization to a certain functional form and keeping track of the determinant of the Jacobian, the authors show that this is still a valid MCMC proposal.

In the second half of the paper, the authors use a neural network to "learn" an optimal transition, where "optimal" may be defined flexibly. I propose integrating the [research implementation](https://github.com/brain-research/l2hmc) built with `tensorflow` into `PyMC3`, making it available for more widespread use.

## Steps

- [x] **Generalize the HMC machinery in `PyMC3`**: The first step will be creating an `L2HMC` class to store the auxiliary functions and handle the bookkeeping for the Jacobian determinant, as well as implementing a modified leapfrog integrator to do this integration. At this point, a test should be implemented that confirms setting all these auxiliary functions to be identically 0 gives the same results as the current `HMC`. It may also be interesting here to run some experiments with different hand-coded auxiliary functions to get an intuition for what they do.

    **Risks** Much of the current code in `PyMC3` is fairly optimized, which makes it harder to extend. In particular, heavy use of the deep learning library `theano` to provide automatic gradients can make passing functions difficult.

    **Notes** There was a nice `CPULeapfrogIntegrator` to work with.  I replaced some quite efficient `BLAS` calls to `axpy` with higher level `numpy` calls, which could probably be replaced with `gemv` later. I ended up implementing it as two functions instead of 6, where the functions are expected to return a 3-tuple.

- [ ] **Implement the training from the paper** In the paper, it is suggested that the six auxiliary functions could be neural networks. The training code is implemented by the authors on [GitHub](https://github.com/brain-research/l2hmc), so this step would involve making a reasonably flexible API for adjusting the neural network parameters, and making their training routing work `PyMC3`.

    **Risks** I am most worried about this step, and passing functions from `theano` to `tensorflow`. It may be possible to run the leapfrog steps in pure Python, or allow them to run in `tensorflow`. The trainng might also be implemented directly in `theano`, but then I lose the ability to follow the authors' implementation as closely.

- [ ] **Run experiments using `L2HMC`** At this point, it should be straightforward to reproduce the experiments from the authors' paper, as well as any other model that can be defined using `PyMC3`.

- [ ] **Open pull request to merge changes** The core developers at this point will make suggestions regarding the API, experiments (contributions such as this usually come with examples, which the above experiments will hopefully take care of), and implementation. At this point I would also get in touch with the authors of the paper.

    **Risks** This is a major contribution, and it is unclear how long it would take other contributors to check it for correctness and quality.

## Further Work

There are a lot of basic questions about the neural network this project will not address. A design goal of `PyMC3` is to specify a model, and have the library automatically determine the best way to do inference. I plan to have a default network architecture and training routine that works for the experiments, but it would be interesting to do something more clever in optimizing this. A simple first step would be reporting diagnostics to warn the user if the neural net failed to converge, and recommending the NUTS sampler instead.