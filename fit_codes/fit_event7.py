from fit_application import fit_light_curve


i=int(7)

path_ephemerides = '/home/anibal/files_db/james_webb.txt'
path_save = '/home/anibal/roman_rubin/event_7_analisys/DE/'
path_model = '/home/anibal/files_db/full_curves/' #PATH OF THE DIRECTORY OF THE LIGHT CURVE THAT I WANT TO FIT

fit_light_curve(path_model+f'Event_{i}.txt', 'TRF', path_ephemerides,path_save,path_model)


