# Function to check if the difference between dates is more than 2 hours
from datetime import datetime, timedelta
import json
import math
import os
import pickle
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import pandas as pd
from consts import Consts,statusCodes
from tslearn.metrics import cdist_dtw ,dtw_path, dtw_limited_warping_length, dtw_subsequence_path
from tslearn.utils import to_time_series_dataset
import folium
from pyproj import CRS, Transformer


def fix_data(dataset1):

    cameras_info=pd.read_csv(Consts['camera_path'])
    # Merge dataset1 with relevant columns from cameras info dataset
    merged_data = pd.merge(dataset1, cameras_info[['ExternalCameraId', 'x', 'y']], 
                       left_on='externalCameraId', right_on='ExternalCameraId', 
                       how='left', suffixes=('', '_gt'))
     # Update the x and y coordinates in dataset1
    mask = merged_data['x_gt'].notna()  # Create a mask where 'x_gt' is not NaN (i.e., a match was found)
    dataset1.loc[mask, 'x'] = merged_data.loc[mask, 'x_gt']
    dataset1.loc[mask, 'y'] = merged_data.loc[mask, 'y_gt']
    
     # Assuming we want to swap 'X' and 'Y' where 'X' < 34.9 -> this condition is ampaircal after doing eda on the csv.
    mask = dataset1['x'] < 34.9
     # Swapping the values
    dataset1.loc[mask, ['x', 'y']] = dataset1.loc[mask, ['y', 'x']].values
    #filter out places where x cordainte is equal to y cordinate
    
    # Create a mask where x is not equal to y-to drop places where is x is qeual to y assuming problem with data.
    mask = dataset1['x'] != dataset1['y']

    # Filter the DataFrame using the mask
    dataset1 = dataset1[mask]
    dataset1.dropna(subset=['x', 'y'])
    return dataset1

def create_voyages(df):
    def diff_greater_than_n_hours(time1, time2,hours=2):
        return (time2 - time1) > timedelta(hours=hours)


    def diff_greater_than_n_sec (time1, time2,seconds=3):
        return (time2 - time1) > timedelta(seconds=seconds)

        # Convert 'date' column to datetime objects
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort the DataFrame by 'LpId' and 'date'
    df.sort_values(by=['LpId', 'Date'], inplace=True)

    # Initialize a dictionary to hold the voyages
    voyages_dict = {}
    # Iterate over the DataFrame, grouped by 'LpId'
    for lpid, group in df.groupby('LpId'):
        voyages = []  # List to hold voyages for this 'LpId'
        current_voyage = []  # List to hold the current voyage's data points
        last_date = None  # Variable to keep track of the last date in the current voyage
        
        for index, row in group.iterrows():
            time_obj =  datetime.strptime(row['Hour'], '%I:%M:%S %p')
            combined_datetime = row['Date'].replace(hour=time_obj.hour, minute=time_obj.minute, second=time_obj.second)
            row['Date']=combined_datetime
            if last_date is not None:
                if diff_greater_than_n_hours(last_date, row['Date'], 2):
                    if len(current_voyage) >= Consts['min_len_voyage']:
                        voyages.append(current_voyage)
                    current_voyage = []
            if last_date is None or diff_greater_than_n_sec(last_date, row['Date'], Consts['min_time_between_samples']):
                if not np.isnan( row['x']) and not np.isnan(row['y']):
                    current_voyage.append([row['x'], row['y'], row['Date']])
            last_date = row['Date']
            # Add the last voyage if it's not empty
        if current_voyage and len(current_voyage) >= Consts['min_len_voyage']:
            voyages.append(current_voyage)
        if voyages:
            voyages_dict[lpid] = voyages
    return voyages_dict

def extract_lps_from_clustters_path(path):
    filenames=os.listdir(path)
    # Regular expression to find the required part
    pattern = re.compile(r'(.*?)\.json')
    # Extracting the required part from each filename in the list
    lp_to_file_path = {pattern.search(filename).group(1):filename if pattern.search(filename) else None for filename in filenames}

    return lp_to_file_path

# Function to determine if a sublist contains any NaN values
def contains_any_nan(sublist):
    return any(np.isnan(item) if isinstance(item, float) else False for item in sublist)


def filter_nans(clusters):
    '''Removes NaN values from the clusters for a given LP.'''
    filtered_clusters = clusters.copy()
    for lp_key, lp_value in filtered_clusters.items():
        for cluster_key, cluster_list in lp_value['Clusters'].items():
            filtered_clusters[lp_key][cluster_key] = [
                sublist for sublist in cluster_list if not any(np.isnan(item) if isinstance(item, float) else False for item in sublist)
            ]
    return filtered_clusters

def read_files(base_path, file_paths):
    '''Reads JSON files from a list of paths and returns their contents.'''
    file_contents = []
    for path in file_paths:
        try:
            with open(os.path.join(base_path, path), 'r', encoding='utf-8') as file:
                content = json.load(file)
                file_contents.append(content)
        except FileNotFoundError:
            print(f"File not found: {path}")
        except Exception as e:
            print(f"An error occurred while reading {path}: {str(e)}")
    return file_contents

def voyages_split_prediction(voyages ,clustters_content,dict_name_of_voyages="inffered_voyages.pkl",utm_format=False):
    """
    Processes a list of voyages of diffrent lps into a structured dictionary and call to function to calcuate
    predction for each voyage base on the closest clusster.

    :param voyages: List of voyages where each voyage is a list of nodes in the format (x, y, date).
    :return: Dictionary structured to contain tail data and other inferred data.
    """
    
    voyages_data  = {lp: [] for lp in clustters_content.keys()}
    for lp in voyages_data.keys():    
        for voyage in voyages[lp]:
            # Sort the voyage by date to ensure correct start and end dates-should be already sorted
            voyage = sorted(voyage, key=lambda x: x[2])           
            start_date = voyage[0][2]
            end_date = voyage[-1][2]          
            # Simulate splitting the voyage into gt_data and inferred data parts
            inferred_data_list = []
            lp_clusters=clustters_content[lp]
            lp_clusters=[cluster for cluster in lp_clusters['Clusters'].values()]
                # time_in_hours=time_obj.hour+time_obj.minute / 60
            
            voyage=[[node[0],node[1],node[2].hour+node[2].minute/60+node[2].second/3600] for node in voyage]
            # till len(voyage) and not len(voyage)+1 couse we want to run until one before the last element
            
            # if lp in lp_to_debug:
            debug_path=os.path.join(Consts['trail_path'],'debug_all.txt') 
            log_lp_result(lp,voyage,lp_clusters,debug_path)
            for tail_splits_len in range(Consts['min_len_voyage_tail'],len(voyage)):
                voyage_tail=voyage[:tail_splits_len]
                res= infer_future_nodes(voyage_tail,lp_clusters)
                if res["status"]==statusCodes.fit_to_Routine:
                    inferred_data= res["infferd_data"]
                    len_inferd=inferred_data.shape[0]
                    weight=res['weight']
                    dtw_metric=res['dtw']
                    best_matched_cluster_idx=res['best_matched_cluster_idx']
                    if utm_format:
                        inferred_data=reshape_cordinates(inferred_data)
                        gt_infferd=reshape_cordinates(voyage[tail_splits_len:])
                    else :
                        gt_infferd=voyage[tail_splits_len:]
                    inferred_data_list.append({
                        'tail_len':tail_splits_len,                            
                        'infferd_data':inferred_data,
                        'gt_infferd':gt_infferd,
                        "weight":weight,
                        "dtw": dtw_metric,
                        "len_inferd":len_inferd,
                        'best_matched_cluster_idx':best_matched_cluster_idx
                    })
                elif  res["status"]==statusCodes.not_fit_to_Routine:
                      inferred_data_list.append({
                        'tail_len':tail_splits_len,
                        'infferd_data':[],
                        'gt_infferd':"not infferable voyage!",
                        "weight":None,
                        "dtw": None,
                        'best_matched_cluster_idx':None
                    })
            if utm_format:  
                voyage=reshape_cordinates(voyage)
            voyage_entry = {
                'start_date': start_date,
                'end_date': end_date,
                'gt_data': voyage,
                'infer_data': inferred_data_list,
                'max_inffered_len':  max((inferred_data_for_tail.get("len_inferd", 0) for inferred_data_for_tail in inferred_data_list))
            }
            voyages_data[lp].append(voyage_entry)
        voyages_data[lp].sort(key=lambda x:x['max_inffered_len'], reverse=True)
    # Sorting the dictionary keys based on 'max_inferred_len' of the first dictionary in each list
    sorted_lps = sorted(voyages_data, key=lambda k: voyages_data[k][0]['max_inffered_len'], reverse=True)
    # Creating a new dictionary with sorted keys
    voyages_data = {k: voyages_data[k] for k in sorted_lps}
    save_dict_to_pickle(voyages_data,dict_name_of_voyages)
    return voyages_data


def infer_future_nodes(voyage,lp_clusters):
    # extract the values from the clussters dict
    lp_clusters =to_time_series_dataset(lp_clusters)
    #if there isnt beyond tail 
    cluster_with_potinel_to_inffer=[filter_uninfferable_clustters(voyage,lp_cluster) for lp_cluster in lp_clusters]
    if not any(cluster_with_potinel_to_inffer)==True:
         return  {"status": statusCodes.not_fit_to_Routine} 
    clusters_with_potinel_to_inffer_idxs = [index for index, value in enumerate(cluster_with_potinel_to_inffer) if value]
    lp_clusters=lp_clusters[clusters_with_potinel_to_inffer_idxs]
    lp_cluster_shaped=[filter_nodes(voyage,lp_cluster,"filter") for lp_cluster in lp_clusters]
    voyage= np.array(voyage)
    voyage=voyage[np.newaxis, :, :]
    cur_dtw = cdist_dtw(voyage,lp_cluster_shaped, n_jobs=1)
    # np.where(cur_dtw<Consts["th_dtw"] and lp_clusters.shape[0]>Consts.min_len_voyage)

    best_matched_cluster_idx = np.argmin(cur_dtw, axis=1)[0]
    # remove the extra axis 
    voyage=voyage[0]
    infferd_data=filter_nodes(voyage,lp_clusters[best_matched_cluster_idx],'predict')
    infferd_data= np.array([list(sublist) for sublist in infferd_data if not contains_any_nan(sublist)])
    weight  = 1/(1+math.exp((1.5*Consts["th_dtw"])-(math.sqrt(cur_dtw[0][best_matched_cluster_idx]))))
    return { "status": statusCodes.fit_to_Routine,"dtw":cur_dtw,"weight":weight,"infferd_data":infferd_data,"best_matched_cluster_idx":best_matched_cluster_idx}


def save_dict_to_pickle(data_dict, filename):
    """
    Saves a dictionary to a JSON file.
    
    :param data_dict: Dictionary to be saved.
    :param filename: Filename for the JSON file.
    """
    try:
        with open(filename, 'wb') as pickle_file:
          pickle.dump(data_dict, pickle_file)
        print(f"Dictionary saved successfully to {filename}")
    except Exception as e:
        print(f"Failed to save dictionary: {str(e)}")
        

def load_dict_from_pickle(filename):
    """
    Loads a dictionary from a JSON file.
    
    :param filename: Filename of the JSON file to be read.
    :return: Loaded dictionary, or None if an error occurs.
    """
    try:
        with open(filename, 'rb') as pickle_file:
           loaded_dict = pickle.load(pickle_file)
        print(f"Dictionary loaded successfully from {filename}")
        return loaded_dict
    except FileNotFoundError:
        print(f"No file found with the name {filename}. Please check the filename and try again.")
    except Exception as e:
        print(f"Failed to load dictionary: {str(e)}")
    return None


def extract_clustters(voyages_lps):
    '''extract clustters from base directorty - according to given list of lps that we have vouage for them 
    '''
    # list the names of file in given directory, return a dict with lps as keys and path value
    clusters_info =extract_lps_from_clustters_path(Consts['clustter_path'])
    # fillter out the cluster path that are not in the tested voyages list  ,return touple with lp , file_path,file_size 
    relevant_clusters = [
        (lp, clusters_info[lp], get_file_size(Consts['clustter_path'], clusters_info[lp]))
        for lp in voyages_lps if lp in clusters_info
    ]
    # Convert to a NumPy array for easier manipulation
    cluster_array = np.array(relevant_clusters, dtype=object)

    # Sort the array by file sizes (third element of each tuple) in descending order
    sorted_clusters = cluster_array[np.argsort(cluster_array[:, 2].astype(int))[::-1]]

    # Now you can extract sorted lists
    lps = sorted_clusters[:, 0]
    file_paths = sorted_clusters[:, 1]
        
    clustters_content= read_files(Consts['clustter_path'],file_paths)
    #fillter nans frm files
    clustters_content=[filter_nans(clustter_content)  for clustter_content in clustters_content]
    # clustters_content=[clustter_content for clustter_content in clustters_content]
    clustters_content={list(clustter_content.keys())[0] : list(clustter_content.values())[0] for clustter_content in clustters_content}
    # fillter out the cluster path that are not in the tested voyages list 
    return clustters_content
    
# def filtter_out_short_clustter():
#     ''' this function filters clusster that are qual or shorten than minmal'''

def euclidean_distance_np(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.sqrt(np.sum((point1 - point2) ** 2))

def filter_nodes(voyage, clustter ,mode="filter"):
    # ''this function filter out nodes from the voyages and cluttter, for cases where there is mismatch in-length 
    #   side note:leagndry implention of this function called searchinpath'''
    deletef=[]
    path=dtw_path(voyage,clustter)[0]
    # search if there are nodes in the path where the first elemnt( that is node in the voyage) or the second elemnt( that is node in the cluster)
    # is reptecd in subsequant elements of the path and is not the last node in voyage
    
    idx_of_last_voyge_node=len(voyage)-1
    for idx in range (len(path)-1):
        if path[idx][0] ==  path[idx+1][0] and path[idx][0]!= (len(voyage)-1):
              deletef.append(idx)    
    updated_path=np.delete(path,deletef , axis=0)
    first_ouccrance_of_last_node_of_voyage=np.where(updated_path[:,0]== idx_of_last_voyge_node)[0][0]
    if mode=="filter":
        #num of nodes to tak in the clustter - take the 
        idx=updated_path[:first_ouccrance_of_last_node_of_voyage+1].T[1][-1]
        return clustter[:idx+1]
    elif mode =="predict":
        # +1 becouse we take the nodes after the one that fit to the last node of the vouage 
        infferd_data_idx=updated_path[first_ouccrance_of_last_node_of_voyage+1:].T[1][0]
        return clustter[infferd_data_idx:]

def filter_uninfferable_clustters(voyage, clustter ):
    deletef=[]
    path=dtw_path(voyage,clustter)[0]
    
    idx_of_last_voyge_node=len(voyage)-1
    for idx in range (len(path)-1):
        if path[idx][0] ==  path[idx+1][0] and path[idx][0]!= (len(voyage)-1):
              deletef.append(idx)    
    updated_path=np.delete(path,deletef , axis=0)
    first_ouccrance_of_last_node=np.where(updated_path[:,0]== idx_of_last_voyge_node)[0][0]
    infferiable_clustter=updated_path[first_ouccrance_of_last_node+1:].size!=0
    return infferiable_clustter



def save_dict_to_json(voyage_data,name):
    def convert(x):
        if hasattr(x, "tolist"):  # Convert numpy arrays to lists
            return x.tolist()
        elif isinstance(x, datetime):  # Convert datetime objects to string
            return x.isoformat()
        raise TypeError(f"Object of type {type(x).__name__} is not JSON serializable")

    # Convert all datetime objects to strings within the dictionary structure
    if(voyage_data!= None):
        for lp, lists_dict_of_tails in voyage_data.items():
            for idx, dict in enumerate(lists_dict_of_tails):
                for key, value in dict.items():
                    if isinstance(value, datetime):
                        voyage_data[lp][idx][key] = value.isoformat()

    # Saving the dictionary to a JSON file
    with open(name, 'w') as file:
        json.dump(voyage_data, file, default=convert)
        

def get_file_size(dir_path,filename):
      return os.path.getsize(os.path.join(dir_path, filename))
  
def log_lp_result(lp,voyage,lp_clusters,file_path='log_results.txt'):
    # Open the file in append mode
    with open(file_path, 'a') as file:
        file.write(f"lp is {lp}\n")
        file.write(f"len lp is {len(voyage)}\n")
        file.write(f"voyage is \n{voyage}\n\n")
        
        for idx, cluster in enumerate(lp_clusters):
            dtw_result = dtw_path(voyage, cluster)  # Calculate DTW once and reuse the result
            file.write(f"cluster {idx} is \n{cluster}\n")
            file.write(f"dtw with all voyage with cluster {idx} is \n{dtw_result}\n\n")
        
        for idx_tail in range(Consts['min_len_voyage_tail'], len(voyage)):
            file.write("\n")  # Print a newline to separate sections
            voyage_tail=voyage[:idx_tail] 
            for idx, cluster in enumerate(lp_clusters):
                dtw_result = dtw_path(voyage_tail, cluster)  # Calculate DTW again for the tail
                file.write(f"dtw with tail, len {idx_tail}, with cluster {idx} is \n{dtw_result}\n")
                
            res= infer_future_nodes(voyage_tail,lp_clusters)
            if res["status"]==statusCodes.not_fit_to_Routine:
                file.write(f"not fit to Routine")
            else:
                file.write(f"ok!! \n  dtw:{res['dtw']} \n infferd_data:{ res['infferd_data']} \n best_match_clustter_idx:{ res['best_matched_cluster_idx']},\n  len_inferd:{res['infferd_data'].shape[0]} \n\n")


def reshape_cordinates(coords_array):
    # Function to convert decimal hour to HH:MM format

    def decimal_to_time24(decimal_hour):
        hours = int(decimal_hour)
        decimal=(decimal_hour - hours) * 60
        minutes = int(decimal)
        seconds=int((decimal-minutes)*60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"
     
    def latlon_to_utm(longitude,latitude):
        # Define the CRS for WGS84
        wgs84_crs = "EPSG:4326"
        utm_zone = '36'
        hemisphere_suffix = '6'
        utm_crs = f"EPSG:32{hemisphere_suffix}{utm_zone}"
        # Create a transformer that converts from WGS84 to the appropriate UTM zone
        transformer =  CRSTransformerSingleton(wgs84_crs, utm_crs)

        # Perform the transformation
        easting, northing = transformer.transform(longitude, latitude)
        northing=int(northing)
        easting= f"{utm_zone}/ {int(easting)}"
        return easting, northing
     # Create a new list to store the modified coordinate tuples
    updated_coords = []

    # Process each coordinate array in the input list
    for coords in coords_array:
        # Extract the last element as the time, convert it, and replace it
        *coordinates, time_decimal = coords
        if np.isnan( coordinates[0]) or np.isnan( coordinates[1]):
            print('im here ')
        formatted_time = decimal_to_time24(time_decimal)
        easting, northing = latlon_to_utm(*coordinates)

        # Append the tuple of transformed coordinates and formatted time
        updated_coords.append((easting, northing, formatted_time))

    return updated_coords
   
# def fillter_nodes(voyage, clustter ):
#     '''this function filter out nodes from the voyages and cluttter, for cases where there is mismatch in-length 
#         case the clusster  is longer find the closet one , case voyage is longer do nothing,
#         side note:leagndry implention of this function called searchinpath'''
#     deletef=[]
#     ii = -1
#     path=dtw_path(voyage,clustter)[0]
#     # search if there are nodes in the path where the first elemnt( that is node in the voyage) or the second elemnt( that is node in the cluster)
#     # is reptecd in subsequant elements of the path.
#     same_elements={}
#     deletef=[]
#     for idx in range (len(path)-1):
#         if path[idx][0] ==  path[idx+1][0] :
#             node_idx_voyage=path[idx][0]
#             if path[idx][0] not in same_elements:
#                 same_elements[ node_idx_voyage]= [idx,idx+1]
#             else:
#                 same_elements[ node_idx_voyage][1]=idx+1
#     for same_element in same_elements:
#         #idx for path that corresond the the same voyage node
#         start_idx,end_idx=same_elements[same_element][0],same_elements[same_element][1]
#         cluster_nodes = [clustter[idx_in_path[1]] for idx_in_path in path[start_idx:end_idx+1]] 
#         voyage_node  = voyage[path[start_idx][0]]
#         print(cluster_nodes)
#         print(voyage_node)
#         index_of_min=min(cluster_nodes, key=lambda cluster_node: euclidean_distance_np(cluster_node,voyage_node))
#         print(index_of_min)


#becouse there is date-time in the scheme the json object is not serializeable



class CRSTransformerSingleton:
    _instance = None
    _transformer = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CRSTransformerSingleton, cls).__new__(cls)
        return cls._instance

    def __init__(self, source_crs_str, target_crs_str):
        if not self._transformer:
            # Initialize the transformer only once
            source_crs=CRS(source_crs_str)
            target_crs=CRS(target_crs_str)
            self._transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    def transform(self, x, y):
        return self._transformer.transform(x, y)