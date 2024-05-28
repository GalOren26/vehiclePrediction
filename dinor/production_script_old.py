import copy

import numpy as np
from tslearn.metrics import cdist_dtw
from datetime import datetime
import pyodbc
import time
import json
import os
import requests

lp_current_voyages = {}
all_clusters = {}


def create_clusters_json():
    json_path = 'json'
    result = dict()
    for f1 in os.listdir(json_path):
        with open(os.path.join(json_path, f1), 'r') as infile:
            res = json.load(infile)
            result.update(res)

    return result






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
    response=requests.post('http://alapp3.al.local/AIAnomalyAlertsTesterFM/api/Data/SendDetection',data=json_data , headers = _headers)
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

        data  = { 'lp': lp , 'detectionId':detectionId, 'alertId':id, 'weight':weight, 'write_voyage':write_vc[:,1:].tolist()  }

        try:
            sendJson('', data)
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
def production_api(logger,DetectionId,lp, x, y, t, total_clusters):
    lp = str(lp)
    th_voyage = 7200
    th_dtw = 1.5

    try:
        if lp in total_clusters:
            clusters = total_clusters[lp]['Clusters']
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
        if sample[-1] - lp_current_voyages[lp][-1][-1] > th_voyage:  # over the maximum threshold of voyage (time)
            curr_voyage = [sample]
            lp_current_voyages[lp] = curr_voyage

        else:  # another sample of the current voyage
            curr_voyage = lp_current_voyages[lp]
            curr_voyage.append(sample)
            lp_current_voyages[lp] = curr_voyage

    # process the voyage before prediction
    curr_voyage = np.array(curr_voyage, dtype='object')
    start_point = curr_voyage[0, -1]
    curr_voyage[:, -1] -= start_point
    curr_voyage[:, -1] /= 10000

    cur_dtw = cdist_dtw(([curr_voyage[:,1:]]) ,np.flip( clusters,-1))
    res = np.min(cur_dtw) / len(curr_voyage)
    if res >= th_dtw:
        # weight  = 1/(1+math.exp((2*th_dtw)-(math.sqrt(res))))
        weight = 1 / (1 + math.exp((1.5 * th_dtw) - res))
        curr_voyage[:, -1] *= 10000
        curr_voyage[:, -1] += start_point
        cv=copy.deepcopy(curr_voyage)
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

