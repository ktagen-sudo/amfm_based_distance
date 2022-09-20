#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from scipy.ndimage.morphology import distance_transform_edt 
import numpy as np
import pandas as pd
from skimage import data, util, measure, morphology


def _generate_dist(_mask):
    
    _edt, _inds = distance_transform_edt(_mask!=0, 
                           return_indices=True)
    
    return _edt, _inds

    
def _lesion_stats(_lesion):
    _mask = _lesion > 0
    
    #We clean the lesion volume of a very small lesions
    _mask = morphology.remove_small_objects(_mask, min_size=20) 
    _label_lesion = measure.label(_mask, connectivity=_lesion.ndim)

    #
    #props = measure.regionprops(_label_image)
    _properties = ['label', 'area', 
                  'centroid', 'axis_major_length', 
                  'axis_minor_length']

    props_table = measure.regionprops_table(_label_lesion,
                           properties=_properties)


    print("{0}".format(80 * "-"))
    print("Printing corresponding properties for that lesion")
    print("{0}".format(80 * "-"))

    _props_table_df = pd.DataFrame(props_table)  
    
    print(_props_table_df)
    
    return _props_table_df, _label_lesion

def _testFun1d(_z_list):
    # Building the test function of the form: 
    #--> exp(-(1 / (10 - x) (10 - x))) * exp(- 1 / (x *x))
    
    #z = np.linspace(z_list[0], z_list[1], num=num_samples)
    #z = np.array(_z_list)
    z = _z_list
    
    #Y[Y==0] = 1.0e-6
    #Z[Z==0] = 1.0e-6
     
    # Main function
    #f = 2.8 * np.exp(-(10 - z) / (10 - z)) * np.exp(-1 / z**2)   
    f = np.exp(-1 / (z**2)) 

    f_params_1d = [f, z]
    
    return f_params_1d