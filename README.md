# QuestionnaireFastTransform

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MagineZ.github.io/QuestionnaireFastTransform.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MagineZ.github.io/QuestionnaireFastTransform.jl/dev/)
[![Build Status](https://github.com/MagineZ/QuestionnaireFastTransform.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MagineZ/QuestionnaireFastTransform.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MagineZ/QuestionnaireFastTransform.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MagineZ/QuestionnaireFastTransform.jl)

## Python Dependencies

This package requires Python packages:

- `numpy`

By default, PyCall uses your system Python (`/usr/bin/python3`). Make sure `numpy` is installed:

```bash
python3 -m pip install numpy
```
Alternatively, in Julia

ENV["PYTHON"] = ""
using Pkg
Pkg.build("PyCall")
using Conda
Conda.add("numpy")
