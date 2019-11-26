import ipykernel
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
