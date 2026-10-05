#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MANIFEST="$SCRIPT_DIR/fenics-spack.yaml"

if [ -z "${SPACK_BIN:-}" ]; then
    if [ -n "${SPACK_ROOT:-}" ]; then
        SPACK_BIN="$SPACK_ROOT/bin/spack"
    else
        SPACK_BIN=$(command -v spack || true)
    fi
fi

if [ -z "${SPACK_BIN:-}" ] || [ ! -x "$SPACK_BIN" ]; then
    printf '%s\n' "Set SPACK_ROOT or SPACK_BIN to a Spack installation." >&2
    exit 2
fi

if [ -z "${SPACK_ENV_DIR:-}" ]; then
    if [ -n "${SPACK_ROOT:-}" ]; then
        SPACK_ENV_DIR="$SPACK_ROOT/var/spack/environments/fem-varden-flow"
    else
        SPACK_ENV_DIR="$HOME/.spack/environments/fem-varden-flow"
    fi
fi

mkdir -p "$SPACK_ENV_DIR"
SPACK_ENV_DIR=$(CDPATH= cd -- "$SPACK_ENV_DIR" && pwd)

if [ ! -f "$SPACK_ENV_DIR/spack.yaml" ]; then
    cp "$MANIFEST" "$SPACK_ENV_DIR/spack.yaml"
elif ! cmp -s "$MANIFEST" "$SPACK_ENV_DIR/spack.yaml"; then
    printf '%s\n' \
        "The environment already exists with a different spack.yaml:" \
        "  $SPACK_ENV_DIR/spack.yaml" \
        "Set SPACK_ENV_DIR to a new directory to install from this recipe." >&2
    exit 2
fi

"$SPACK_BIN" -e "$SPACK_ENV_DIR" concretize
"$SPACK_BIN" -e "$SPACK_ENV_DIR" install

eval "$("$SPACK_BIN" env activate --sh "$SPACK_ENV_DIR")"
python - <<'PY'
import dolfin
import numpy
from dolfin import FunctionSpace, TestFunction, TrialFunction, UnitSquareMesh, assemble, dx

mesh = UnitSquareMesh(2, 2)
function_space = FunctionSpace(mesh, "P", 1)
mass_matrix = assemble(TrialFunction(function_space) * TestFunction(function_space) * dx)

print("NumPy:", numpy.__version__)
print("DOLFIN:", dolfin.__version__)
print("Smoke-test matrix:", mass_matrix.size(0), "x", mass_matrix.size(1))
PY
