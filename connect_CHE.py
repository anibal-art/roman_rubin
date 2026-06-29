from scp import SCPClient
from tqdm.auto import tqdm
import os


def download_data(ssh, sources_array, nset, path_run, path_save):

    for source in tqdm(sources_array):
        path_true = path_run + f'/set_sim{nset}/Event_{source}.h5'
        path_rr = path_run + f'/set_fit{nset}/Event_RR_{source}_TRF.npy'
        path_roman = path_run + f'/set_fit{nset}/Event_Roman_{source}_TRF.npy'
        
        os.makedirs(path_save, exist_ok=True)
        
        with SCPClient(ssh.get_transport()) as scp:
            scp.get(path_rr, path_save)
            scp.get(path_roman, path_save)
            scp.get(path_true, path_save)
        
        print('done')
