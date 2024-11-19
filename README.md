# pdp-2025
The code used for the development of the paper

#### Singularity
The first step in reproducing this environment is to build the singularity container. Run the following command in the project root directory
```
singularity build --fakeroot singularity/singularity.sif singularity/singularity.def
```
Once the container has been created, access it with the following command:
```
singularity shell --nv singularity/singularity.sif
```
Use --nv when using gpu, this flag enables NVIDIA GPU support.