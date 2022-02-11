#!/bin/sh -eu

PYTHON="python"
PY_EXP_VER_MAJ="3"
PY_EXP_VER_MIN="6"

PYV_MAJ=$(${PYTHON} -c "import sys; print(list(sys.version_info[:2])[0])";)
PYV_MIN=$(${PYTHON} -c "import sys; print(list(sys.version_info[:2])[1])";)

print_and_error_exit() {
  echo "${1}"
  exit 1
}

if [ "${PYV_MAJ}" -lt "${PY_EXP_VER_MAJ}" ]; then
   print_and_error_exit "Expected Python >=v${PY_EXP_VER_MAJ}.${PY_EXP_VER_MIN}, and detected $(${PYTHON} -V)"
fi
if [ "${PYV_MIN}" -lt "${PY_EXP_VER_MIN}" ]; then
   print_and_error_exit "Expected Python >=v${PY_EXP_VER_MAJ}.${PY_EXP_VER_MIN}, and detected $(${PYTHON} -V)"
fi

THIS_SCRIPT_DIR="$(cd $(dirname ${0}); pwd)"
PYTHON_VENV_PATH="${THIS_SCRIPT_DIR}/.pyvenv"
REQ_FILE="${THIS_SCRIPT_DIR}/Requirements.txt"

${PYTHON} -m pip install --upgrade pip

status=0
${PYTHON} -m venv ${PYTHON_VENV_PATH} || status=$?
if [ "${status}" -ne 0 ]; then
    is_pyenv_installed=0
    ${PYTHON} -m pip list | grep pipenv &>/dev/null || is_pyenv_installed=$?
    if [ ${is_pyenv_installed} -ne 0 ]; then
        ${PYTHON} -m pip install pipenv
        ${PYTHON} -m venv ${PYTHON_VENV_PATH}
    fi
fi
set +u
. ${PYTHON_VENV_PATH}/bin/activate
set -u

python -m pip install -r ${REQ_FILE}
python -m pip install --upgrade setuptools

echo
echo "Python virtual environment is created at - ${PYTHON_VENV_PATH}"
echo
exit 0
