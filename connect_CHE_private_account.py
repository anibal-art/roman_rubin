from paramiko import SSHClient, AutoAddPolicy
from scp import SCPClient
from tqdm.auto import tqdm
import os
def download_data(sources_array, nset, path_run, path_save):

    ssh = SSHClient()
    ssh.set_missing_host_key_policy(AutoAddPolicy())
    ssh.connect(
        hostname='152.84.248.250', 
        port=13900,  # <-- tu puerto personalizado
        username='anibalvarela', 
        password='38victorioso177'
    )
    for source in tqdm(sources_array):
        # source = 147
        #nset = 1
        
        # BH  FFP  Planets_systems
        #system = {'USBL':'Planets_systems','FSPL':'FFP','PSPL':'BH'}
        #path_run = f'/share/storage3/rubin/microlensing/romanrubin/RR2025/Baseline4_set/'+system[model]
        
        path_true = path_run + f'/set_sim{nset}/Event_{source}.h5'
        path_rr = path_run + f'/set_fit{nset}/Event_RR_{source}_TRF.npy'
        path_roman = path_run + f'/set_fit{nset}/Event_Roman_{source}_TRF.npy'
        
        #path_save = '../baseline4/'+model+'/'
        
        os.makedirs(path_save, exist_ok=True)
        
        with SCPClient(ssh.get_transport()) as scp:
            scp.get(path_rr, path_save)
            scp.get(path_roman, path_save)
            scp.get(path_true, path_save)
        
        print('done')
