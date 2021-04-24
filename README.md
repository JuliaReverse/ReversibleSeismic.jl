# ReversibleSeismic

Automatic differentiating PML solver in seismic simulation.
For examples, see https://github.com/GiggleLiu/WuLiXueBao

[![CI](https://github.com/JuliaReverse/ReversibleSeismic.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/JuliaReverse/ReversibleSeismic.jl/actions/workflows/ci.yml)

## Features

* Optimal checkpointing (Treeverse algorithm),
* Reversible programming (Bennett algorithm) implemented in [NiLang.jl](https://github.com/GiggleLiu/NiLang.jl)),
* Differentiating CUDA kernel functions with [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) + NiLang.jl.

It can differentiate a PML solver defined on a 5000 x 5000 grid with 10000 time steps on a 32G memory GPU in one hour.

## References

* https://github.com/kailaix/ADSeismic.jl/
* https://github.com/geodynamics/seismic_cpml
* Efficient PML for the wave equation [arxiv: 1001.0319](https://arxiv.org/abs/1001.0319)

If you want to credit this repo, please cite
[not published yet]()
