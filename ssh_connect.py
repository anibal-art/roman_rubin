from paramiko import SSHClient, AutoAddPolicy

def ssh_che():
    ssh = SSHClient()
    ssh.set_missing_host_key_policy(AutoAddPolicy())
    ssh.connect(
        hostname='152.84.248.250', 
        port=13900,  # <-- tu puerto personalizado
        username='anibalvarela', 
        password='38victorioso177'
    )
    return ssh
