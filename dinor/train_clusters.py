import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
import collections
import json
import pyodbc
import time
import os

from multiprocessing import Process
from matplotlib import pyplot as plt
from tslearn.metrics import cdist_dtw
from tslearn.clustering import silhouette_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
debug = False
# from ALAI_api import *



def create_voyages(time_values, x_val, y_val, threshold_voyage):
    curr_voyage = []
    lp_voyages = []
    for index, (h, x, y) in enumerate(zip(time_values, x_val, y_val)):
        t_i = datetime(int(time_values[index][6:10]),
                                int(time_values[index][3:5]),
                                int(time_values[index][:2]),
                                int(time_values[index][11:13]),
                                int(time_values[index][-5:-3]),
                                int(time_values[index][-2:]))
        if index == 0:
            before = [ x, y ,t_i]
            if before not in curr_voyage:
                curr_voyage.append([ x, y , t_i])

        elif (t_i - before[2]).total_seconds() <= threshold_voyage:
            if [ x, y,t_i] not in curr_voyage:
                curr_voyage.append([ x, y,t_i])
                before = [ x, y,t_i]

        else:
            before = [ x, y,t_i]
            lp_voyages.append(curr_voyage)
            curr_voyage = [before]
    lp_voyages.append(curr_voyage)
    return lp_voyages


def create_valid_voyages(voyages):
    """
    Creating the 3D data (n,k,3) where:
        n - The number of roads per voyage.
        k - The number of samples per road, where k lies in the range: [min_length_voyage, max_length_voyage]
        3 - (t,x,y)

    valid_voyages - contains the numpy array of the whole data where the time is described as timedata type.
    valid_voyages_integer - contains the numpy array of the whole data where the time is described as integer.
    """
    good_voyages = []
    for s_voyage in voyages:
        s_voyage = np.array(s_voyage)
        t_axis = s_voyage[:, 2]
        t_axis_int = np.array([(t.hour + ( t.minute / 60)) for t in t_axis])

        #t_axis_int_shifted = (t_axis_int - t_axis_int[0])/10000
        x_axis_val = s_voyage[:,0]
        y_axis_val = s_voyage[:, 1]

        new_arr = np.vstack(( x_axis_val, y_axis_val,t_axis_int)).T
        _,idx=  np.unique(new_arr.astype('float64'), axis=0 , return_index=True)
        new_arr=new_arr.astype('float64')[np.sort(idx)]
        good_voyages.append(new_arr)

    else:
        return np.array(good_voyages, dtype='object')

    
def input_data(LP_csv):
    hour_values, date_values = LP_csv['Hour'], LP_csv['Date']
    LP_csv['Total_Date'] = date_values + ' ' + hour_values
    return LP_csv['x'], LP_csv['y'], LP_csv['Total_Date']


def creating_valid_voyages(LP_csv, min_len_voyage, max_len_voyage, th_voyage):
    x_values, y_values, date_time_values = input_data(LP_csv)
    date_time_values_list = date_time_values.tolist()
    datetime_list_sorted = date_time_values_list.copy()
    #datetime_list_sorted.sort(key=lambda date: datetime.strptime(date, "%d/%m/%Y %H:%M:%S"))
    #sort_index = [j for i in range(len(date_time_values)) for j in range(len(date_time_values)) if date_time_values_list[i] in datetime_list_sorted[j]]

    x_values_sort = x_values
    y_values_sort = y_values

    '''Creating the voyages array'''
    LP_voyage = create_voyages(datetime_list_sorted, x_values_sort, y_values_sort, th_voyage)
    valid_voyages = np.array([np.array(voyage) for voyage in LP_voyage
                              if min_len_voyage <= len(voyage) <= max_len_voyage], dtype='object')

    return valid_voyages


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
    save_figures_dir = 'figures results'
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


def save_test_fig(lp, best_matched_cluster, test_voyages, best_clusters, dtw_distance , show=False):
    save_figures_dir = 'figures results'
    mksureDir(save_figures_dir)
    if lp not in os.listdir(save_figures_dir):
        lp_fig_path = os.path.join(save_figures_dir, str(lp)+'_3d')
        if not os.path.exists(lp_fig_path):
            os.mkdir(lp_fig_path)
        for idx, test_voyage in enumerate(test_voyages):
            save_bestC_fig(lp, best_clusters)
            cluster_idx = best_matched_cluster[idx]
            paired_cluster = best_clusters[cluster_idx]
            ax = plt.axes(projection='3d')

            yaxistrack = [ round(elem,2)  for elem in
                              test_voyage[:, 1]]



            ax.plot3D(paired_cluster[:, 0].astype('float64'), paired_cluster[:,1].astype('float64'), paired_cluster[:, 2].astype('float64'), '-bo', label="cluster")
            ax.plot3D(test_voyage[:, 0].astype('float64'),   yaxistrack , (test_voyage[:, 2].astype('float64')), '-ro', label="test")
            #plt.plot(paired_cluster[:,1], paired_cluster[:,2],'-r',label="cluster")
            #plt.plot(test_voyage[:,1], test_voyage[:,2], 'b', label="Test voyage")
            #plt.xlabel('x axis')
            #plt.ylabel('y axis')
            #plt.title(f"Test voyage vs paired cluster for lp: {lp} and dtw: {dtw_distance[idx]}")
            #plt.legend()
            ax.set_title(f"Test voyage vs paired cluster for lp: {lp} and dtw: {dtw_distance[idx]}")
            ax.legend()
            ax.set_zlabel("time (int value)")
            ax.set_xlabel("X value")
            ax.set_ylabel("Y value")
            fig_path = os.path.join(lp_fig_path, f"img{idx}_3d_{dtw_distance}_.jpg")
            plt.savefig(fig_path)
            plt.clf()
            ax.clear()



def save_test_fig_run(lp, best_matched_cluster, test_voyages, best_clusters, dtw_distance , show=False):
    save_figures_dir = 'figures results'
    mksureDir(save_figures_dir)
    if lp not in os.listdir(save_figures_dir):
        lp_fig_path = os.path.join(save_figures_dir, str(lp)+'_3d')
        if not os.path.exists(lp_fig_path):
            os.mkdir(lp_fig_path)
        for idx, test_voyage in enumerate(test_voyages):
            save_bestC_fig(lp, best_clusters)
            cluster_idx = best_matched_cluster[idx]
            paired_cluster = best_clusters[cluster_idx]
            ax = plt.axes(projection='3d')

            yaxistrack = [ round(elem,2)  for elem in
                              test_voyage[:, 1]]



            ax.plot3D(paired_cluster[:, 0].astype('float64'), paired_cluster[:,1].astype('float64'), paired_cluster[:, 2].astype('float64'), '-bo', label="cluster")
            ax.plot3D(test_voyage[:, 0].astype('float64'),   yaxistrack , (test_voyage[:, 2].astype('float64')), '-ro', label="test")
            #plt.plot(paired_cluster[:,1], paired_cluster[:,2],'-r',label="cluster")
            #plt.plot(test_voyage[:,1], test_voyage[:,2], 'b', label="Test voyage")
            #plt.xlabel('x axis')
            #plt.ylabel('y axis')
            #plt.title(f"Test voyage vs paired cluster for lp: {lp} and dtw: {dtw_distance[idx]}")
            #plt.legend()
            ax.set_title(f"Test voyage vs paired cluster for lp: {lp} and dtw: {dtw_distance}")
            ax.legend()
            ax.set_zlabel("time (int value)")
            ax.set_xlabel("X value")
            ax.set_ylabel("Y value")
            fig_path = os.path.join(lp_fig_path, f"img{idx}_3d_{dtw_distance}_.jpg")
            plt.savefig(fig_path)
            plt.clf()
            ax.clear()


def predict_test_voyage(train_voyages, test_voyages, best_n_cluster, th_anomaly_dtw, lp):
    labels = TimeSeriesKMeans(n_clusters=best_n_cluster,
                                 metric="dtw").fit_predict(train_voyages)

    # Removing the noisy cluster (count less than one precent of the train set)
    count_labels = collections.Counter(np.array(labels))
    num_train_voyages = len(train_voyages)
    good_labels = []
    for state in count_labels:
        if count_labels[state]/num_train_voyages > 0.01:
            good_labels.append(state)

    best_clusters = find_best_clusters(train_voyages, labels, good_labels)
    save_bestC_fig(lp,   best_clusters )
    # Predicting the result (Anomaly vs Normal) for the test set
    test_dtw_matrix = cdist_dtw(test_voyages, best_clusters)
    min_dtw_distance = np.min(test_dtw_matrix, axis=1)
    best_matched_cluster = np.argmin(test_dtw_matrix, axis=1)

    #save_test_fig(lp, best_matched_cluster, test_voyages, best_clusters,min_dtw_distance)

    return np.array(min_dtw_distance > th_anomaly_dtw), best_matched_cluster, best_clusters


def get_lp_samples_from_sql(lp_number):
    # Connect to the SQL server
    # conn = pyodbc.connect('DRIVER={SQL SERVER};'
    #                               'SERVER=db2\eldb;'
    #                               'UID=ALAISvc;'
    #                               'PWD=Al123456#1;')

    conn = pyodbc.connect('DRIVER={SQL SERVER};'
                          'SERVER=db1\eldb;'
                          'UID=ALAISvc;'
                          'PWD=eoya11Afrodithe1!')

    cursor = conn.cursor()
    query = """SELECT
          [LpId]
          ,Location.STX as x
    	  ,Location.STY as y
    	  ,LEFT(EventDate,10) as Date
    	  ,RIGHT(EventDate,12) as Hour
          ,[LpTypeId]
      FROM [ALProd22].[dbo].[Detections] 
      WHERE EventDate > '2023-08-01'   and EventDate < '2023-12-19'      
      and Location.STX is not Null and Location.STY is not Null 
      and LpId= '""" + str(lp_number)+ """' and 1 =? order by EventDate asc """

    start_time = time.time()
    cursor.execute(query, 1 )

    rows = cursor.fetchall()

    df_lp = pd.DataFrame.from_records((tuple(t) for t in rows))
    df_lp.columns = [column[0] for column in cursor.description]
    df_lp['Hour'] = [df_lp['Hour'][i][:8] for i in range(df_lp.shape[0])]
    df_lp['Date'] = [datetime.strftime(datetime.strptime(date, '%Y-%m-%d'),"%d/%m/%Y") for date in df_lp['Date']]

    return df_lp


def isTimeFormat(input):
    try:
        time.strptime(input,'%Y-%m-%d')
        return True
    except ValueError:
        return False


def train_function(lpidx=0):
    global splitted_list
    lp_data = str(splitted_list[lpidx])
    process_num = os.getpid()
    try:
        """Constant Variables"""
        th_voyage = 7200  # max time distance between two samples in seconds (= 2 hours) - 7200
        min_length_voyage = 3 # min length of a voyage
        max_length_voyage = 50 # max length of a voyage
        th_anomaly_dtw = 1.5 # predefined threshold for anomaly detection (dtw distance)
        final_cluster_dict = {}

        N = len(str(lp_data))
        running_time = time.time()
        count = 0
        lp = lp_data

        matches = [match for match in all_clusters if '_' + lp + '.json' in match]


        if len(matches) == 1:
            print(str(lp)+" Exist....")
            if not debug:
                return
            else:
                os.remove(f'json/{matches[0]}')
        print(str(lp) + " Build....")
        lp_cluster_dict = {}

        lp_csv = get_lp_samples_from_sql(lp)    

        # Arranging constant format for date and hour
        lp_csv['Hour'] = [lp_csv['Hour'][i][:8] for i in range(lp_csv.shape[0])]
        if isTimeFormat(lp_csv['Date'][0]):
            lp_csv['Date'] = [datetime.strftime(datetime.   strptime(date, '%Y-%m-%d'), "%d/%m/%Y") for date in lp_csv['Date']]

        valid_voyages = creating_valid_voyages(lp_csv,
                                               min_length_voyage,
                                               max_length_voyage,
                                               th_voyage)

        valid_voyages_integer = create_valid_voyages(valid_voyages)  # Contains the valid voyages (int time values)

        if  valid_voyages_integer.shape[0] > 30  :

            count += 1
            train_voyages_arr, test_voyages_arr = train_test_split(to_time_series_dataset(valid_voyages_integer),
                                                                   test_size=0.2,shuffle=True)
            # idxlst = [np.any(i) for i in np.isnan(train_voyages_arr)]
            # train_voyages_arr = np.delete(train_voyages_arr,idxlst, axis=0)
            low = 1
            high = 10
            best_cluster_number = optimal_parameters_score(train_voyages_arr,
                                                           low, high)

            # Evaluate the test voyages
            anomaly_prediction, best_matching_clusters, best_clusters = predict_test_voyage(train_voyages_arr,
                                                                                            test_voyages_arr,
                                                                                            best_cluster_number,
                                                                                            th_anomaly_dtw,
                                                                                            lp)
            cluster_dict = {k: v.tolist() for k, v in enumerate(best_clusters)}
            lp_cluster_dict["Clusters"] = cluster_dict
            accuracy = 1-np.sum(anomaly_prediction)/len(anomaly_prediction)
            print(process_num, lp , "accu :" + str(accuracy))

            lp_cluster_dict["test accuracy"] = accuracy*100
            final_cluster_dict[lp] = lp_cluster_dict
            try:
                json_path = "C:/production/api/josn/clusters" + str(process_num+200) +"_"+ str(lp)+".json"
            except:
                exit(-1)
            with open(json_path, "w") as fp:
                json.dump(final_cluster_dict, fp, indent=3)
            final_cluster_dict = {}

            if count % 20 == 0:
                print(f"Process {process_num} Completed cluster calculation of {count}/{N} license plates")
                print(f"Running time for process {process_num} is seconds from last sampling: {time.time()-running_time}")
                running_time = time.time()
        else:
            print(lp_csv.shape[0] > 1000 and valid_voyages_integer.shape[0] > 0 and 5 < len(str(lp)))

    except Exception as ex:
        print (ex)
        pass

from multiprocessing import Pool, TimeoutError

lps_path = 'count_lp.csv'


lps_list = pd.read_csv(lps_path)[0:]
splitted_list = np.array(lps_list['LpId'])

if __name__ == "__main__":



    # r =12
    # splitted_list = np.array_split(np.array(lps_list['LpId']), r)

    with Pool(processes=12) as pool:
        pool.map(train_function ,  range (splitted_list.__len__()))
        #train_function(index, splitted_list[index])


        # p = Process(target=train_function, args=(index, splitted_list[index],))
        # p.start()




