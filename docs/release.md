## Stable Releases
Currently we follow a lightweight release process.
- Update the version number in `assets/version.txt` with a PR. The version numbering should follow https://semver.org/.
  - E.g. for a pre-release `0.y.z`
    - if major features are added, increment `y`
    - if minor fixes are added, increment `z`
- Create a new release at https://github.com/pytorch/torchtitan/releases/new
  - In the tag section, add a new tag for the release. The tag should use the version number with a `v` prefix (for example, `v0.1.0`). Make sure to select the `main` branch as the target.
  - In the release notes
    - include proper nightly versions for `torch` and `torchao` if those are part of the release notes. E.g.
        - "Successfully installed ... `torch-2.8.0.dev20250605+cu126`"
        - "Successfully installed `torchao-0.12.0.dev20250605+cu126`"
    - describe the release at a high level compared to the last release, e.g.
      - "added an experiment for multimodal LLM training"
      - or simply state "this is a regular release"
  - For now, choose "Set as a pre-release".
- Package publishing is not automated in this repository state. If a PyPI update is needed, publish it manually.

The general instruction on managing releases can be found [here](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository).


## Nightly Builds
Nightly wheel publication is not automated in this repository state.
