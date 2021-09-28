# GATE-PUBLIC

## Requirements

*  A CUDA capable GPU. You can verify if you have one by checking in device manager for a display adapter that is an NVIDIA product. Then check if your product is CUDA enabled [here](https://developer.nvidia.com/cuda-gpus#compute).
*  A CUDA compiler (this repo has been tested on Windows with nvcc CUDA 10.1 and 10.2). No guarantees for other CUDA versions. Linux is currently not supported, but will be in the near future.
*  MSVC 2017 or 2019 with C/C++ tool and CMake integration. (See Notes 1.)

### Notes

1.  Information about CMake integration with Microsoft Visual Studio can be found for [2017](https://docs.microsoft.com/en-us/cpp/build/cmake-projects-in-visual-studio?view=vs-2017) and for [2019](https://docs.microsoft.com/en-us/cpp/build/cmake-projects-in-visual-studio?view=vs-2019)
2.  Download CUDA from Nvidia's [website](https://developer.nvidia.com/cuda-downloads)

## Building and Running

Open the repository folder in Microsoft Visual Studio. The CMakeCache should automatically be generated, then you should be able to build any of the cmake targets in Release configuration. Debug is currently not supported.

To verify the build, run any of the examples in \gate-public\gate\examples, e.g., gate_quad_main.exe.

## Usage

See examples in \gate-public\gate\examples for usage. A user is required to create a dynamics class, a controller class, and a perturbations class. See our tutorial paper for further details:

Gandhi, M., Schlossman, R., Williams, K.A., Melzer, R., & Parish, J. (2021). CUDA for Rapid Controller Robustness Evaluation: A Tutorial. 

## Contributing

If you are interested in contributing, please contact rschlos@sandia.gov. 

## Acknowledgements

We acknowledge Jason Gibson and Bogdan Vlahov for their help in developing the base code structure.

## License

[License](LICENSE)

Sandia National Laboratories is a multimission laboratory managed and operated by National Technology & Engineering Solutions of Sandia, LLC, a wholly owned subsidiary of Honeywell International Inc., for the U.S. Department of Energy’s National Nuclear Security Administration under contract DE-NA0003525.


## Copyright
Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

# Notice
For five (5) years from 9/28/2021 the United States Government is granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable worldwide license in this data to reproduce, prepare derivative works, and perform publicly and display publicly, by or on behalf of the Government. There is provision for the possible extension of the term of this license. Subsequent to that period or any extension granted, the United States Government is granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable worldwide license in this data to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so. The specific term of the license can be identified by inquiry made to National Technology and Engineering Solutions of Sandia, LLC or DOE.
 
NEITHER THE UNITED STATES GOVERNMENT, NOR THE UNITED STATES DEPARTMENT OF ENERGY, NOR NATIONAL TECHNOLOGY AND ENGINEERING SOLUTIONS OF SANDIA, LLC, NOR ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY INFORMATION, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.
 
Any licensee of this software has the obligation and responsibility to abide by the applicable export control laws, regulations, and general prohibitions relating to the export of technical data. Failure to obtain an export control license or other authority from the Government may result in criminal liability under U.S. laws.
