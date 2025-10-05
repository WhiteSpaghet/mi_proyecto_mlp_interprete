import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from mlp_numpy import MLP, Layer
from interprete import compile_model
from entrenar import model, history