import GPUtil
import json 
import numpy as np
import tensorflow as tf


def tfconstant(x):
        return tf.constant(x, dtype=tf.float64)

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

def print_GPUinfo():
  GPUtil.showUtilization(attrList=[[{'attr':'memoryUtil','name':'Memory util.','suffix':'%','transform': lambda x: x*100,'precision':0}],
                        [{'attr':'memoryTotal','name':'Memory total','suffix':'MB','precision':0},]])

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)