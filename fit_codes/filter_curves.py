import pandas as pd
import numpy as np
def binning_curves(event,fil, n):
    '''
    binning curves
    '''
    event_sorted = event[event['band']==fil].sort_values(by='mjd')
    fs = event_sorted['mjd'].values[0]
    tin = event_sorted['mjd'].values[0]
    tfin = evento_sorted['mjd'].values[-1]
    bins = np.arange(tin,tfin+n,n)
    evento_sorted['binned'] = pd.cut(event_sorted['mjd'], bins,labels=False)
    time = []
    mags = []
    errmags = []
    for i in range(len(bins)):
        if not len(event_sorted[event_sorted['binned']==i])==0:
            time.append(np.mean(event_sorted[event_sorted['binned']==i]['mjd']))
            mags.append(np.mean(event_sorted[event_sorted['binned']==i]['mag']))
            errmags.append(np.sqrt(sum(np.array(event_sorted[event_sorted['binned']==i]['magerr'])**2))/len(event_sorted[event_sorted['binned']==i]['magerr']))
    df0 = pd.DataFrame({
        'band': [fil]*len(errmags),
        'mjd': time,
        'mag': mags,
        'magerr': errmags})
    return df0

def chi2(mag,err,c):
    '''
    Function to compute chi squared respect to a baseline magnitude 
    '''
    l = []
    for i in range(len(mag)):
        l.append((mag[i]-c)**2/err[i]**2)
    return np.sum(l)/len(mag)

def filtros(file_name):
    '''
	List of selected criteria for filtering events where the peak is detected:

	- Only consider points that are within 1 sigma distance from the limiting magnitude.
	- Only consider events that have observations in both the Roman and Rubin bands.
	- Require a minimum of 10 data points in any given filter.
	- Consider points that fall outside the time range of t0 ± 5tE; compute their mean value to establish the magnitude baseline.
	- Calculate the chi-squared value for the entire light curve, assuming a constant baseline magnitude model as previously computed.
	- If the chi-squared value is > 2 for any filter, consider it as an event.
    '''
    evento = pd.read_csv(file_name , sep = ',' , decimal = '.', skiprows = 19)
    params = pd.read_csv(file_name , sep = ':' , decimal = '.', skiprows = 0)
    
    filtercolor = {'w':'b','u':'c', 'g':'g', 'r':'y', 'i':'r', 'z':'m', 'y':'k'}
    mag_sat = {'w':14.8, 'u':14.7, 'g': 15.7, 'r': 15.8, 'i': 15.8, 'z': 15.3, 'y': 13.9}
    
    dict_params = {}
    for i in range(len(params[0:17].values[:,1])):
        dict_params[params[0:17].values[:,0][i]] = params[0:17].values[:,1][i]

    t0 = dict_params['t0']
    dict_params['te'] = dict_params.pop('tE')
    te = dict_params['te']
    
    #------------if has mag over mag_sat and if it's more than 1 sigma (error bar)----------------
    evento.loc[evento['band'] == 'w', 'm5'] = 25.9
    evento['mag_sat'] = evento['band'].map(mag_sat)
    criteria = evento['m5'] > evento['mag'] + 1*evento['magerr']
    criteria2 = evento['mag']>evento['mag_sat']
    filtered_evento = evento[criteria&criteria2]
    evento = filtered_evento
    
    #----if the band have less than 4 pints is not considered----------------    
    value_counts = evento['band'].value_counts()   #cuenta el número de filas(datos) en cada banda
    # Get the bands with less than 4 values
    classes_to_drop = value_counts[value_counts < 6].index.tolist()
    # Drop the classes with less than 4 values from the DataFrame
    evento = evento[~evento['band'].isin(classes_to_drop)]
    evento = evento.groupby('band').filter(lambda group: len(group) >= 10) #chequea que existan al menos 10 puntos por banda

    #----------------Extraction of the main part of the event for the binning -----------------------------------
    only_event = evento.loc[(evento['mjd'] < t0+3.5*te) & (evento['mjd'] > t0-3.5*te)]
    only_event = only_event[['band','mjd','mag','magerr']]
    interval = [t0-5*te,t0+5*te]
    no_event = evento[~evento['mjd'].between(interval[0], interval[1])]
    #---------------If there are no Rubin bands the event is not considered-----------------------
    is_only_w = (evento['band'] == 'w').all()
    #We apply the chi_squared criteria
    if not is_only_w:
        crit_1 = {}
        chis=[]
        for fil in ('w','u','g','r','i','z','y'):
            if fil in evento['band'].values:
                    df = evento
                    mjd = df[df['band']==fil][['mjd','mag','magerr']].values[:,0]
                    mag = df[df['band']==fil][['mjd','mag','magerr']].values[:,1]
                    magerr = df[df['band']==fil][['mjd','mag','magerr']].values[:,2]
                    crit_1[fil] = np.c_[mjd,mag,magerr]
                    mag_baseline = np.mean(no_event['mag'][no_event['band']==fil])
                    chis.append(chi2(mag,magerr,mag_baseline))
            else:
                crit_1[fil] = np.array([])
        is_value_greater_than_2 = False
        for value in chis:
            if value > 2:
                is_value_greater_than_2 = True
                break  # No need to continue checking once we find one value greater than 2

        if is_value_greater_than_2:
            for fil in ('w','u','g','r','i','z','y'):
                if fil in evento['band'].values:
                        df = evento
                        mjd = df[df['band']==fil][['mjd','mag','magerr']].values[:,0]
                        mag = df[df['band']==fil][['mjd','mag','magerr']].values[:,1]
                        magerr = df[df['band']==fil][['mjd','mag','magerr']].values[:,2]
                        crit_1[fil] = np.c_[mjd,mag,magerr]
            return crit_1, dict_params
        else:
            return np.array([]),np.array([])
    else:
        return np.array([]),np.array([])
    
    
    
def no_filtros(file_name,tit, N,binning):
    '''
    function with binnig
    Falta el filtro de eliminar las curvas que no tienen bandas de Rubin
    '''
    evento = pd.read_csv(file_name , sep = ',' , decimal = '.', skiprows = 20)
    params = pd.read_csv(file_name , sep = ':' , decimal = '.', skiprows = 0)
    
    filtercolor = {'w':'b','u':'c', 'g':'g', 'r':'y', 'i':'r', 'z':'m', 'y':'k'}
    mag_sat = {'w':14.8, 'u':14.7, 'g': 15.7, 'r': 15.8, 'i': 15.8, 'z': 15.3, 'y': 13.9}
    
    dict_params = {}
    for i in range(len(params[0:18].values[:,1])):
        dict_params[params[0:18].values[:,0][i]] = params[0:18].values[:,1][i]
#     print(dict_params)

    t0 = dict_params['t0']
    dict_params['te'] = dict_params.pop('tE')
    te = dict_params['te']
    
    #------------SI TIENE MAG MENOS DE LA MAG_SAT Y SI ESTA MAS DE 1 SIGMA----------------
    evento.loc[evento['band'] == 'w', 'm5'] = 25.9
    evento['mag_sat'] = evento['band'].map(mag_sat)
    criteria = evento['m5'] > evento['mag'] + 1*evento['magerr']
    criteria2 = evento['mag']>evento['mag_sat']
    filtered_evento = evento[criteria&criteria2]
    evento = filtered_evento
    
    crit_1 = {}
    for fil in ('w','u','g','r','i','z','y'):
        if fil in evento['band'].values:
            df = evento
            mjd = df[df['band']==fil][['mjd','mag','magerr']].values[:,0]
            mag = df[df['band']==fil][['mjd','mag','magerr']].values[:,1]
            magerr = df[df['band']==fil][['mjd','mag','magerr']].values[:,2]
            crit_1[fil] = np.c_[mjd,mag,magerr]

        else:
            crit_1[fil] = np.array([])
                
    return crit_1, dict_params
    
    
    
