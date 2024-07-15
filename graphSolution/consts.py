from enum import Enum
Consts={
'clustter_path':'.\\train\\trained_clustters',
'th_dtw':14,
"min_len_voyage_tail":2,
"min_len_voyage":2,
"min_time_between_samples":3 ,# in seconds
"camera_path":"..\\AllLprCameras.csv",
"trail_path":"..\\gal\\trails_lat_lon\\all",
"detections_path_name":"GAL_Det-08-2023-ALL-AI.csv"
}
class statusCodes(Enum):
    not_fit_to_Routine = 0
    fit_to_Routine = 1
class Color(Enum):
    RED = 1
    ORANGE = 2
    YELLOW = 3
    GREEN = 4
    BLUE = 5
    INDIGO = 6
    VIOLET = 7
    PINK = 8
    BROWN = 9
    BLACK = 10
