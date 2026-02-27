To accelerate contributions to and innovations around `torchtitan`, we are adding this new, experimental folder. Below are the general contributing guidelines, and we look forward to your contributions!

## Contributing Guidelines

We provide this `experiments/` folder to host experiments that add significant value to `torchtitan`, with the following principles. We refer to the part of `torchtitan` outside `experiments` as `core`.
1. Each subfolder in `experiments` will be an experiment, with a clear theme which can be flexible, such as
    - A new model, or preferably a new model architecture, with its training infrastructure including parallelization functions. Please see the [instructions](/torchtitan/models/README.md) on how to contribute a new model.
    - An enhancement or addition to the existing infrastructure of `torchtitan`.
2. It is the contributors' responsibility to justify the value of an experiment. `torchtitan` team will review proposals on a case-by-case basis. As part of the contribution, the contributors should provide documentation that clearly showcases the motivation and innovation of an experiment, including reports on performance and loss convergence.
3. An experiment should reuse existing `torchtitan` code as much as possible, such as modules in [`components/`](../components/) (via a new [`ModelSpec`](../protocols/model_spec.py)) and [`train.py`](../train.py). For a list of extension points we provide, please refer to [docs/extension.md](../../docs/extension.md).
    - The extension points are subject to change. We kindly request that contributors provide feedback if they encounter issues reusing any components, rather than simply using a copy-and-paste approach.
    - The degree to which existing components are reused and whether duplications are legit will also be a criteria of whether an experiment would be accepted.
4. Each experiment is independent from other experiments, and can have its own dependencies (on top of [core dependencies](../../requirements.txt)), and its own tests. An experiment should not contain vendor-specific code, such as kernels written in a proprietary language. Those can be hosted outside as dependency.
5. The dependency from `experiments` to `core` is one-way. Anything in `experiments` is optional for `core` to run successfully. In particular, development in `core` is not blocked by breakage in `experiments`. Experiment owners are expected to validate changes with local tests for the experiment and any touched `core` paths.
6. Each experiment needs to have an owner. The owner is responsible to work with `torchtitan` team to maintain the quality and healthiness of an experiment, which includes
    - adapting an experiment to changes in `core` and fix broken tests, no later than the next official `torchtitan` release;
    - responding to GitHub issues and questions in a timely manner.
7. `torchtitan` team reserve the right to remove an experiment. In particular, an experiment should be removed if
    - it has served its purpose (e.g., providing findings, or getting some features upstreamed to `core` or PyTorch, etc.), or
    - it gets stale (e.g. not being maintained).


## Current experiments

| Experiment | Owners |
| ----- | ----: |
| [simple_fsdp](./simple_fsdp/) | [@ruisizhang123](https://github.com/ruisizhang123) [@tianyu-l](https://github.com/tianyu-l) |
| [compiler_toolkit](./compiler_toolkit/) | [@SherlockNoMad](https://github.com/SherlockNoMad) [@yiming0416](https://github.com/yiming0416) |
| [autoparallel](./autoparallel/) | [@wconstab](https://github.com/wconstab) [@xmfan](https://github.com/xmfan) |
| [torchcomms](./torchcomms/) | [@d4l3k](https://https://github.com/d4l3k) [@fduwjj](https://github.com/fduwjj) [@mori360 ](https://github.com/mori360) |
| [ft](./ft/) | [@tushar00jain](https://github.com/tushar00jain) [@fegin](https://github.com/fegin) |
| [vlm](./vlm/) | [@lkhphuc](https://github.com/lkhphuc) [@shuhuayu](https://github.com/shuhuayu) |
| [transformers_modeling_backend](./transformers_modeling_backend/) | [@3outeille](https://github.com/3outeille) |
| [rl](./rl/) | [@wwwjn](https://github.com/wwwjn) |
| [forge](./forge/) | [@allenwang28](https://github.com/allenwang28) [@joecummings](https://github.com/joecummings) [@felipemello1](https://github.com/felipemello1) [@daniellepintz](https://github.com/daniellepintz) |
