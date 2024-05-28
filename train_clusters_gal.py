import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
import collections
import json
import time
import os

from multiprocessing import Process
from matplotlib import pyplot as plt
from tslearn.metrics import cdist_dtw
from tslearn.clustering import silhouette_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from utils import create_voyages,fix_data
debug = False
#from ALAI_api import *

def optimal_parameters_score(voyages_int, min_n, max_n):

    silhouette_score_k_means = []
    n_array = []
    start_time = time.time()
    start_time_per_lp = time.time()
    for n_cluster in range(min_n, max_n):
        try:
            hc_labels = TimeSeriesKMeans(n_clusters=n_cluster,
                                         metric="dtw").fit_predict(voyages_int)
            ################ S SCORE ! ! ################
            if 1 < len(np.unique(hc_labels)) < voyages_int.shape[0]:
                sil_score_hier_clustering = silhouette_score(voyages_int,
                                                             hc_labels,
                                                             metric="dtw")
                n_array.append(n_cluster)
                silhouette_score_k_means.append(sil_score_hier_clustering)
        except:
            continue

    if not n_array: # n_array = []
        return min_n

    else:
        n_array = np.array(n_array)
        idx = np.where(np.array(silhouette_score_k_means) > 0.85)
        if np.sum(idx) == 0:
            optimal_clusters_num = n_array[np.argmax(silhouette_score_k_means)]
        else:
            optimal_clusters_num = n_array[idx][-1]

        return optimal_clusters_num


def find_best_clusters(data, total_labels, valid_clusters_index):
    """
    Finds the best clusters for data and labels of specific model.
    :param data: numpy array.
    Contains the 3D data of voyages.
    :param y_predict: numpy array.
    Contains the prediction of a specific model on the data.
    :return: numpy array
    Optimal clusters where the index represent the number of the best cluster of specific class.
    """
    optimal_clusters = []
    for n in valid_clusters_index:
        total_n_clusters = data[np.where(total_labels == n)]
        distance_matrix = cdist_dtw(total_n_clusters)
        min_cluster_index = np.argmin(np.sum(distance_matrix, axis=1))
        optimal_clusters.append(total_n_clusters[min_cluster_index])
    return optimal_clusters

def mksureDir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def save_bestC_fig(lp,  best_clusters):
    save_figures_dir = 'figures results new'
    mksureDir(save_figures_dir)
    if lp not in os.listdir(save_figures_dir):
        lp_fig_path = os.path.join(save_figures_dir, str(lp) + '_3d')
        if not os.path.exists(lp_fig_path):
            os.mkdir(lp_fig_path)
        ax = plt.axes(projection='3d')
        for clust in best_clusters:
            ax.plot3D(clust[:, 0], clust[:, 1], clust[:, 2] )

        # plt.plot(paired_cluster[:,1], paired_cluster[:,2],'-r',label="cluster")
        # plt.plot(test_voyage[:,1], test_voyage[:,2], 'b', label="Test voyage")
        # plt.xlabel('x axis')
        # plt.ylabel('y axis')
        # plt.title(f"Test voyage vs paired cluster for lp: {lp} and dtw: {dtw_distance[idx]}")
        # plt.legend()
        ax.set_title(f"clusters for lp: {lp}")
        ax.legend()
        ax.set_zlabel("time (int value)")
        ax.set_xlabel("X value")
        ax.set_ylabel("Y value")
        fig_path = os.path.join(lp_fig_path, f"clusters_3d.jpg")
        # plt.show(block=False)
        plt.savefig(fig_path)
        plt.clf()
        ax.clear()
def merge_dicts(dict_list):
    """
    Merge a list of dictionaries. If they have the same key, concatenate their values.

    :param dict_list: List of dictionaries to merge
    :return: A single merged dictionary
    """
    merged_dict = {}
    
    for d in dict_list:
        for key, value in d.items():
            if key in merged_dict:
                if isinstance(merged_dict[key], list) and isinstance(value, list):
                    merged_dict[key].extend(value)
                else:
                    merged_dict[key] = [merged_dict[key], value] if not isinstance(merged_dict[key], list) else merged_dict[key] + [value]
            else:
                merged_dict[key] = value
                
    return merged_dict
           
def predict_test_voyage(train_voyages, test_voyages, best_n_cluster, lp):
    labels = TimeSeriesKMeans(n_clusters=best_n_cluster,
                                 metric="dtw").fit_predict(train_voyages)
    # Removing the noisy cluster (count less than one precent of the train set)
    count_labels = collections.Counter(np.array(labels))
    num_train_voyages = len(train_voyages)
    good_labels = []
    for state in count_labels:
        if count_labels[state]/num_train_voyages > 0.01:
            good_labels.append(state)
    # save_bestC_fig(lp,   best_clusters )
    best_clusters = find_best_clusters(train_voyages, labels, good_labels)
    # Predicting the result (Anomaly vs Normal) for the test set
    test_dtw_matrix = cdist_dtw(test_voyages, best_clusters)
    best_matched_cluster = np.argmin(test_dtw_matrix, axis=1)

    return best_matched_cluster, best_clusters

def isTimeFormat(input):
    try:
        time.strptime(input,'%Y-%m-%d')
        return True
    except ValueError:
        return False


# def train_function(lpidx=0):
    # global voyages
    # global count 
    # global N 
def train_function(voyages,lp):
    global count 
    global N 
    count+=1
    running_time = time.time()
    try:
            #here expect to get data of some lp 
        if  len(voyages[lp])> 30  :
            for idx,voyage in enumerate(voyages[lp]):
                voyages[lp][idx]=[[node[0],node[1],node[2].hour+node[2].minute/60+node[2].second/3600] for node in voyage]
            train_voyages_arr, test_voyages_arr = train_test_split(to_time_series_dataset(voyages[lp]),
                                                                   test_size=0.2,shuffle=True)
            low = 1
            high = 10
            best_cluster_number = optimal_parameters_score(train_voyages_arr,low, high)
            # Evaluate the test voyages
            _, best_clusters = predict_test_voyage(train_voyages_arr,test_voyages_arr,best_cluster_number,lp)
            cluster_dict = {k: v.tolist() for k, v in enumerate(best_clusters)}
            lp_cluster_dict={}
            final_lp_cluster_dict={}
            lp_cluster_dict['Clusters']=cluster_dict
            final_lp_cluster_dict[lp]=lp_cluster_dict
            try:
                json_path=os.path.join( r".\train\trained_clustters", str(lp)+".json")
            except:
                exit(-1)
            with open(json_path, "w") as fp:
                json.dump(final_lp_cluster_dict, fp, indent=3)
            if count % 20 == 0:
                print(f"Completed cluster calculation of {count}/{N} license plates")
                print(f"Running time in seconds from last sampling: {time.time()-running_time}")
                running_time = time.time()
                
    except Exception as ex:
        print (ex)
# from multiprocessing import Pool, TimeoutError
# lps_path = 'count_lp.csv'
# lps_list = pd.read_csv(lps_path)[0:]
if __name__ == "__main__":
    directory_path=r".\train"
    csv_files_path= csv_file_paths =os.path.join(directory_path, 'GAL_Det-08-2023-ALL-AI.csv') 
    voyages= create_voyages(fix_data(pd.read_csv(csv_files_path)))
    # voyages=merge_dicts(dfs)
    lps=voyages.keys()
    count=0
    N=len(lps)
    for idx in range (len(lps)):
            lp=list(lps)[idx]
            train_function(voyages,lp)
    # with Pool(processes=12) as pool:
    #     pool.map(train_function ,  range (len(lps)))




 