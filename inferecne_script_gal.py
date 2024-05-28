import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from consts import Consts
from utils import create_voyages, load_dict_from_pickle, save_dict_to_pickle, voyages_split_prediction,extract_clustters,save_dict_to_json,fix_data
import cProfile
import io
import pstats
from tslearn.metrics import dtw_subsequence_path,dtw_path,subsequence_cost_matrix
# profiler = cProfile.Profile()
# profiler.enable() 

def dtw(x, y):
    """
    Compute the Dynamic Time Warping (DTW) distance between two time series x and y,
    where x and y are lists of lists representing points in n-dimensional space.
    Returns the DTW distance and prints the distance matrix.
    
    Parameters:
    - x: list of lists, first time series (n_samples, n_features)
    - y: list of lists, second time series (m_samples, n_features)
    
    Returns:
    - float: The DTW distance between the time series
    """
    # Convert lists to numpy arrays
    x = np.array(x)
    y = np.array(y)

    n, m = len(x), len(y)
    C = np.full((n + 1, m + 1), float('inf'))
    C[0, :] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist = np.sum((x[i-1] - y[j-1]) ** 2)
            C[i, j] = dist + min(C[i-1, j], C[i, j-1], C[i-1, j-1])

    print("Distance Matrix:")
    print(C[1:, 1:])  # Print the distance matrix excluding the initial row and column of infinities
    return C



# detections_path='GAL_Det-08-2023-ALL-AI.csv'
# detections_path=os.path.join(Consts['trail_path'],detections_path) 
# df = pd.read_csv(detections_path)
# df=fix_data(df)
# voyages=create_voyages(df)
# voyages_lps=voyages.keys()
# clustters_content=extract_clustters(voyages_lps)
# # lp_to_debug=["1139539","1000963","1028237","1047456","4405130"]
# inffered_voyages=voyages_split_prediction(voyages,clustters_content)  

# profiler.disable() 
# # Create a stream for the stats and print them
# s = io.StringIO()
# ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
# ps.print_stats()-
# # Assume you print this when hitting a breakpoint
# print(s.getvalue())


clustters_content_path=os.path.join(Consts['trail_path'],'clustters_content.pkl') 
# save_dict_to_pickle(data_dict=clustters_content,filename=clustters_content_path)
#becouse there is date-time in the scheme the json object is not serializeable
dict_name_of_voyages=os.path.join(Consts['trail_path'],'inffered_voyages.pkl') 
json_voyages_path=os.path.join(Consts['trail_path'],'inffered_voyages.json') 


# save_dict_to_json(inffered_voyages,json_voyages_path)
# save_dict_to_pickle(inffered_voyages,dict_name_of_voyages)
inffered_voyages=load_dict_from_pickle(dict_name_of_voyages)
clustters_content=load_dict_from_pickle(clustters_content_path)


x=dtw(inffered_voyages['1039331'][2]['gt_data'][:2],clustters_content['1039331']['Clusters']['2'])
y=dtw_subsequence_path(inffered_voyages['1039331'][2]['gt_data'][:2],clustters_content['1039331']['Clusters']['2'])
print(x)
print(y)
# sort_by_len_of_inferance
# create list len_infered_per_voyage
# max_infferd_for_voyage=max(len_infered_per_voyage)-get max for voyage
# create list of max_infferd_for_voyage
# sorted_voyages=sort voyages by max_infferd_for_voyage
# create list of max_infferd_for_lp from sorted_voyages
# max(max_infferd_for_lp  )


# pick lp 
# put his clustters and voyages on map where there is index for each clustter and other index for each voyage when hovering above the line , the line of the cluster and the voyages are with diffrent color 
# 
