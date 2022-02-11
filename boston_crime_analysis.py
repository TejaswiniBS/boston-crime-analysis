import argparse
import sys
import os

# Append the project src directory
PROJECT_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(f"{PROJECT_ROOT_DIR}/src")

from src.apriori import apriori
from src.dtc import dtc
from src.dtr import dtr
from src.etc import etc
from src.knn import knn
from src.knn_sampling import knn_sampling
from src.lr import lr
from src.rfc_hyper_param import rfc_hyper_param
from src.rfc import rfc
from src.rfr import rfr

#******************************************************************************
DATA_DIR=os.path.join(PROJECT_ROOT_DIR, "data")

MODEL_INVOKES = {
    "apriori": apriori, 
    "dtc":dtc, 
    "dtr": dtr, 
    "etc": etc, 
    "knn":knn, 
    "knn_sampling":knn_sampling, 
    "lr":lr, 
    "rfc":rfc, 
    "rfc_hyper_param":rfc_hyper_param, 
    "rfr":rfr
}

def process_input_args(args):
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='BOSTON Crime Analysis(2015-2021)')

    parser.add_argument('command', choices=MODEL_INVOKES.keys())
    return parser.parse_args()


if __name__ == "__main__":
    args = process_input_args(sys.argv[1:])
    MODEL_INVOKES[args.command](data_dir=DATA_DIR)
