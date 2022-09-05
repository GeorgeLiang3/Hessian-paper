# ## Model Simple Anticline, published in Geophysics. Flat Prior, first review

# import tensorflow as tf
# import nptyping 
# # tf.test.gpu_device_name()
# # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# import sys
# import os
# print(os.getcwd())
# sys.path.append('../GP_old')
# sys.path.append('../models')

# # suppress warinings
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

# # dependency
# import gempy as gp
import theano
print(theano.__version__)
print('Done')