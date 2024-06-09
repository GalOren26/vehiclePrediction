import os
from consts import Consts
import utils

voyages_path=os.path.join(Consts['trail_path'],'inffered_voyages.pkl')
clustters_path=os.path.join(Consts['trail_path'],'clustters_content.pkl')
voyages=utils.load_dict_from_pickle(voyages_path)    
print('im here ') 