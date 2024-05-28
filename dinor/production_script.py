import copy

import tslearn.metrics

from train_clusters import *
import numpy as np
from tslearn.metrics import cdist_dtw , dtw,dtw_path
from datetime import datetime
import time
import json
import os
import requests


result = dict()
lp_current_voyages = {}
all_clusters = {}


def create_clusters_json():
    json_path = 'josn'
    # result = dict()
    flist = os.listdir(json_path)
    # counter = flist.__len__()
    # for f1 in flist:
    #     with open(os.path.join(json_path, f1), 'r') as infile:
    #         res = json.load(infile)
    #         result.update(res)
    #         counter -=1
    #         print(counter)

    return flist





def format_sample(DetectionId,x_val, y_val, t_str):
    t_str = t_str.replace("T", " ")
    t_str = t_str.split('.')[0]

    t_object = datetime.strptime(t_str, '%Y-%m-%d %H:%M:%S')
    sample = [DetectionId,x_val, y_val, t_object.timestamp()]
    return sample


def callstoredProc(conn, lp , detectionId ,weight, anom_type=1):

    sql = """SET NOCOUNT ON
             DECLARE	@return_value int
             EXEC	@return_value = [AL_AI].[dbo].[LpAlertInsert]
                    @Lpid = %s ,  @DetectionId = %s ,@Anom_Type = %s  ,@Anom_Strength = %s 
             SELECT	'Return Value' = @return_value""" % (lp ,detectionId, anom_type, weight)
    cursor = conn.cursor()
    cursor.execute(sql)

    row = cursor.fetchone()[0]
    conn.commit()
    return row


def sendJson( url  , obj):
    json_data = json.dumps(obj)
    upurl = url
    _headers = {'Content-type': 'application/json' , 'Accept':'text/plain'}
    response=requests.post(url,data=json_data , headers = _headers)
    print (response.text)



def write_result_to_table(conn,vc,weight, voyage, lp ,detectionId):
    """
    Writes the anomaly voyage to the db on both of the sql tables
    :param values: ndarray - the anomaly voyage
    :return: none
    """
    cursor = conn.cursor()
    id = callstoredProc(conn, lp , detectionId,weight)
    write_voyage = np.hstack((np.ones((voyage.shape[0], 1))*id, voyage))
    write_vc = np.hstack((np.ones((vc.shape[0], 1)) * id, vc))
    try:
        start_time = time.time()
        cursor.executemany("""INSERT INTO [AL_AI].[dbo].[AIAlertsVoyage] (AlertID, DetectionId, LocX ,LocY,DateTime ) 
                              values (?,?,?,?,?)""", write_voyage.tolist() )
        conn.commit()
        print(f"Writing to table AIAlertsVoyage took: {time.time()-start_time}")

        data  = { 'lp': lp , 'detectionId':detectionId, 'alertId':id, 'weight':weight*100, 'write_voyage':write_vc[:,1:].tolist()  }

        try:
            stop=1
            r=stop
            sendJson('http://ppapp1.ishit.idf/AIAnomalyAlertsFM/aiAlert', data)
        except Exception as extag:
            print(f"Error occurred with")
            print(f"{extag}")
            pass


    except pyodbc.Error as ex2:
        sqlstate = ex2.args[0]
        print(f"SQL Error occurred with state {sqlstate} on table AIAlertsVoyage")
        print(f"Error message {ex2}")

    cursor.close()
    conn.close()

    return id

import math

def searchinpath(path):
    deletef=[]
    ii = -1

    for idx in range (len(path)-1):
        if path[ii][0] ==  path[ii-1][0] or path[ii][1] ==  path[ii-1][1]:
            deletef.append(ii)
            pass
        ii=ii-1
    return np.delete(path,deletef , axis=0)

def production_api(logger,DetectionId,lp, x, y, t, total_clusters):
    lp = str(lp)
    th_voyage = 7200
    th_dtw =14
    matches=[]

    try:
        if lp in result :
            clusters = result[lp]['Clusters']
            #clusters = list(result.values())[0]['Clusters']

            pass
        else:
            matches = [match for match in total_clusters if '_'+lp+'.json'  in match ]
            if  len(matches) ==1 :

                with open('C:/production/api/josn/'+ matches[0], 'r') as infile:
                    res = json.load(infile)
                    result.update(res)
                    clusters = result[lp]['Clusters']
                   # clusters = list(result.values())[0]['Clusters']
            else:
                print(f"lp {lp} does not have clusters")
                logger.warning(f"lp {lp} does not have clusters")
                return 0

    except KeyError as e:
        print(f"lp {lp} does not have clusters")
        return 0
    clusters = np.array(list(clusters.values()))
    sample = format_sample(DetectionId,x, y, t)  # for every relevant lp (pre trained)
    if lp not in lp_current_voyages.keys():  # if there is no voyage saved so for for the specific lp
        curr_voyage = [sample]
        lp_current_voyages[lp] = curr_voyage

    else:  # voyage exists for the tested lp
        if     sample[-1] - lp_current_voyages[lp][-1][-1]  > th_voyage:  # over the maximum threshold of voyage (time)
            curr_voyage = [sample]
            lp_current_voyages[lp] = curr_voyage
        else:  # another sample of the current voyage
            if  sample[-1] - lp_current_voyages[lp][-1][-1] < 3:
                return -1
            curr_voyage = lp_current_voyages[lp]
            curr_voyage.append(sample)
            lp_current_voyages[lp] = curr_voyage

    # process the voyage before prediction
    curr_voyage = np.array(curr_voyage, dtype='object')
    #cp_curr_voyage = copy.deepcopy(curr_voyage)
    #start_point = datetime.utcfromtimestamp(curr_voyage[:, -1])
    curr_voyage[:, -1] =  [((datetime.fromtimestamp(x).hour + ( datetime.fromtimestamp(x).minute / 60))) for x in curr_voyage[:, -1]]


    # test_dtw_matrix = cdist_dtw(test_voyages, best_clusters)
    # min_dtw_distance = np.min(test_dtw_matrix, axis=1)

    if curr_voyage.shape[0]<4:
        return -1
    #cdist_dtw , dtw,dtw_path, dtw_subsequence_path , dtw_limited_warping_length

    cur_dtw = cdist_dtw(([curr_voyage[:,1:]]) ,clusters  , n_jobs=1)
    best_matched_cluster = np.argmin(cur_dtw, axis=1)
    op1=-1
    if clusters[int(best_matched_cluster)].shape[0] > curr_voyage[:,1:].shape[0]:
        validpath = searchinpath( np.array( dtw_path((curr_voyage[:,1:]) ,clusters[int(best_matched_cluster)])[0]))
        op1= dtw ((curr_voyage[:, 1:]), clusters[int(best_matched_cluster)][validpath.T[1]])    
    else:
        op1 = dtw((curr_voyage[:, 1:]), clusters[int(best_matched_cluster)])


    res = op1# np.min(cur_dtw , axis=1)  #/ len(curr_voyage)
    if res >= th_dtw:
        lst_voyages=[]
        lst_voyages.append(np.reshape(curr_voyage,(-1,4))[:, 1:] )

        #best_matched_cluster = np.argmin(cur_dtw, axis=1)
        save_test_fig_run(lp, best_matched_cluster, lst_voyages, clusters, res , False)

        weight  = 1/(1+math.exp((1.5*th_dtw)-(math.sqrt(res))))
        #weight = 1 / (1 + math.exp((1.5 * th_dtw) - res))
        cv=copy.deepcopy(curr_voyage)
        curr_voyage = lp_current_voyages[lp]
        curr_voyage=np.array(curr_voyage, dtype='object')


        if curr_voyage.shape[0] > 1:
            curr_voyage[:, -1] = [datetime.fromtimestamp(int(curr_voyage[i, -1])) for i in range(curr_voyage.shape[0])]
        else:
            curr_voyage[:, -1] = datetime.fromtimestamp(int(curr_voyage[:, -1]))

        conn = pyodbc.connect('DRIVER={SQL SERVER};'
                              'SERVER=db2\eldb;'
                              'UID=ALAISvc;'
                      'PWD=Al123456#1;')
        cur_id = write_result_to_table(conn, cv,weight, curr_voyage, lp , DetectionId)

        return cur_id

    return -1


if __name__ == "__main__":
    lp = 6290485
    x_i = 35.1
    y_i = 32.3
    str_time = '2021-12-01T00:00:04.59'
    production_api(str(lp), x_i, y_i, str_time, all_clusters,1)

