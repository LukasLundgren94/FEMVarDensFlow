#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
BUILD_JOBS=${BUILD_JOBS:-10}

if [ -n "${SPACK_BIN:-}" ]; then
    :
elif [ -n "${SPACK_ROOT:-}" ]; then
    SPACK_BIN="$SPACK_ROOT/bin/spack"
else
    SPACK_BIN=$(command -v spack || true)
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

if [ ! -f "$SPACK_ENV_DIR/spack.yaml" ]; then
    printf '%s\n' "Spack environment manifest not found: $SPACK_ENV_DIR/spack.yaml" >&2
    exit 2
fi

eval "$("$SPACK_BIN" env activate --sh "$SPACK_ENV_DIR")"
export FEM_VARDEN_FLOW_ROOT="$PROJECT_ROOT"
export FEM_VARDEN_FLOW_BIN="$PROJECT_ROOT/bin"

for form_dir in \
    "$PROJECT_ROOT/core/src/ufl/norm" \
    "$PROJECT_ROOT/core/src/ufl/pdesmooth" \
    "$PROJECT_ROOT/varden/src/solver/ufl" \
    "$PROJECT_ROOT/varden/src/common/ufl"
do
    (
        cd "$form_dir"
        python3 generate_ufl.py *.ufl.src
        ./compile_forms *.ufl
    )
done

cmake -S "$PROJECT_ROOT" -B "$PROJECT_ROOT/build"
cmake --build "$PROJECT_ROOT/build" --target install --parallel "$BUILD_JOBS"

cd "$SCRIPT_DIR"
cmake .
cmake --build . --parallel "$BUILD_JOBS"
