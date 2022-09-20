#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib as plt
from scipy.ndimage.morphology import distance_transform_edt 
import numpy as np
import pandas as pd
from skimage import data, util, measure, morphology


def _generate_dist(_mask):
    
    edt, inds = distance_transform_edt(_mask!=0, 
                           return_indices=True)
    
    return edt, inds


def _contour_slices(_slices, cmap="gray", total_levels=6):
    # This function includes also plotting contour plot
    # link --> https://www.python-course.eu/matplotlib_contour_plot.php
    # link --> https://github.com/silx-kit/silx/issues/2242
    
    fig, axes = plt.subplots(1, len(_slices), figsize=(5,5))
    contour_levels = total_levels
    titles = ['sagital', 'coronal', 'axial']
    
    for i, slice in enumerate(_slices):
        row, col = np.shape(slice)
        y = np.arange(0, row)
        x = np.arange(0, col)
        xx, yy = np.meshgrid(x, y)

        
        zzmin, zzmax = np.min(slice), np.max(slice)
        levels = np.linspace(zzmin, zzmax, contour_levels)

        #Display image with contour plot
        #axes[i].imshow(slice.T, cmap=cmap, origin="lower")
        contour = axes[i].contour(yy, xx, slice, levels)
        axes[i].clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=12)
        #c = ('#ff0000', '#ffff00', '#0000FF', '0.6', 'c', 'm')
        axes[i].set_title(titles[i])
        #contour_filled = axes[i].contourf(yy, xx, slice, colors=c)

        
    plt.show()
    
def _show_slices(_slices, cmap="gray"):
    # This function comes from the nibabel
    # tutorial --> https://nipy.org/nibabel/coordinate_systems.html#introducing-someone
    
    fig, axes = plt.subplots(1, len(_slices))

    for i, slice in enumerate(_slices):
        axes[i].imshow(slice.T, cmap=cmap, origin="lower")
        
    plt.show()
    
def _lesion_stats(_lesion):
    _mask = _lesion > 0
    
    #We clean the lesion volume of a very small lesions
    _mask = morphology.remove_small_objects(_mask, min_size=20) 
    _label_image = measure.label(_mask, connectivity=_lesion.ndim)

    #
    #props = measure.regionprops(_label_image)
    properties = ['label', 'area', 
                  'centroid', 'axis_major_length', 
                  'axis_minor_length']

    props_table = measure.regionprops_table(_label_image,
                           properties=properties)


    print("{0}".format(80 * "-"))
    print("Printing corresponding properties for that lesion")
    print("{0}".format(80 * "-"))

    props_table_df = pd.DataFrame(props_table)  
    
    print(props_table_df)
    
    return props_table_df, _label_image