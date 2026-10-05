# Arrhenius FEniCS Installation

## Install FEniCS with Spack
FEMVarDensFlow uses legacy FEniCS 2019.1.0.post0. The tested environment manifest and installer are [fenics-spack.yaml](dardel_help/fenics-spack.yaml) and [install-fenics-spack.sh](dardel_help/install-fenics-spack.sh). The pins below are intentional compatibility constraints:

- NumPy 1.26.4: FEniCS 2019's form compiler calls `numpy.product`, removed in NumPy 2.
- PETSc 3.20.6 and SLEPc 3.20.2: newer PETSc releases changed C++ APIs used by DOLFIN 2019.
- Cython 3.0.12: petsc4py 3.20 requires Cython 3 but conflicts with Cython 3.1 and newer.
- The environment view excludes `py-setuptools` to avoid duplicate files already supplied by Spack's Python venv. The package remains installed as a build dependency.

The manifest registers GCC 11.5.0 at `/usr`. On a different host, confirm that this compiler and its `/usr/bin/gcc`, `/usr/bin/g++`, and `/usr/bin/gfortran` paths exist; adjust the compiler external block in the manifest if they do not. On systems that prohibit builds on login nodes, run the installer inside an allocation or submit it through the site's scheduler.

To install with a Spack checkout, set `SPACK_ROOT` to its directory and run from the project root:

```sh
export SPACK_ROOT=/path/to/spack
sh dardel_help/install-fenics-spack.sh
```

Alternatively, set `SPACK_BIN` to the Spack executable and `SPACK_ENV_DIR` to the desired environment directory. The installer copies the tracked manifest only when that directory has no `spack.yaml`; it refuses to replace a different existing environment file. It then concretizes, installs, activates the environment for a smoke test, and assembles a small finite-element mass matrix.

For subsequent interactive use, activate the environment in each shell:

```sh
. "$SPACK_ROOT/share/spack/setup-env.sh"
spack env activate "${SPACK_ENV_DIR:-$SPACK_ROOT/var/spack/environments/fem-varden-flow}"
```

The activation step sets `DOLFIN_DIR` and the package paths required by DOLFIN's Python interface. A `pkg_resources` deprecation warning from legacy UFL may appear during import; it is non-fatal.
