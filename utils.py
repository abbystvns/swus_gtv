from sklearn import linear_model
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn import preprocessing
import pickle
import os


class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        #print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            #print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            #print("done.", flush=True)
            idx += batch_size


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))


def temporal_split(X, y, t):
    X_train = X[:t]
    X_test = X[t:]
    y_train = y[:t]
    y_test = y[t:]
    return X_train, X_test, y_train, y_test


def compute_errors(y_true, y_predicted):
    r2 = 1 - sum((y_true - y_predicted)**2) / sum((y_true - np.mean(y_true))**2)
    mse = mean_squared_error(y_predicted, y_true)
    return(mse, r2)


def aggregate_by_index(X, chunk_size):
    p = X.shape[1]
    new_X = X[:, :chunk_size].mean(axis=1).reshape(-1,1)
    for i in np.arange(chunk_size, p, chunk_size):
        sX = X[:, i:i+4].mean(axis=1).reshape(-1,1)
        new_X = np.hstack([new_X, sX])

    return(preprocessing.scale(new_X))


def load_predictors(data_source, time_frame='one_month', n=None, month=None):
    if data_source == 'obs':
        # load observational SST
        X = pickle_load('data/X_new_obs.pkl')
        fts = pickle_load('data/fts_new.pkl')

        #LENS
        Xlens = preprocessing.scale(pickle_load('data/Xlens_new.pkl'))
        lens_df = pickle_load('data/lens_df_for_detrending.pkl')

    elif data_source == 'reanalysis':
        # load reanalysis
        X = pickle_load('data/reanalysis_X.pkl')
        Xlens = pickle_load('data/LENS_4_reanalysis.pkl')
        fts = pickle_load('data/reanalysis_fts.pkl')



    if time_frame == 'monthly':
        time_map = {
            'july':0,
            'aug':1,
            'sept':2,
            'oct':3
        }

    elif time_frame == 'seasonal':
        time_map = {
            'july':1,
            'aug':1,
            'sept':2,
            'oct':2
        }
    elif time_frame == 'one_month':
        time_map = {
            'july':1,
            'aug':1,
            'sept':2,
            'oct':2
        }
        ix = fts[fts.month==month].index
        fts = fts[fts.month==month].reset_index()
        X = X[:, ix]
        Xlens = Xlens[:, ix]

    fts['time'] = fts.month.apply(lambda x: time_map[x])
    fts['lat'] = fts.lat.astype(int)
    fts['lon'] = fts.lon.astype(int)
    fts = fts.rename(columns={'var':'variable'})

    if time_frame == 'seasonal':
        X = aggregate_by_index(X, 2)
        Xlens = aggregate_by_index(Xlens, 2)
        fts = fts.drop_duplicates(['lat', 'lon', 'time']).reset_index()

    # process NZI
    #nnzi = fts[fts.nzi>0].time.value_counts().iloc[0]
    #Xnzi = aggregate_by_index(X[:, (fts[fts.nzi>0].sort_values('time')).index], nnzi)
    # process ENSO
    #nenso = fts[fts.enso==3.4].time.value_counts().iloc[0]
    #Xenso = aggregate_by_index(X[:, (fts[fts.enso==3.4].sort_values('time')).index], nenso)
    #Xtele = np.hstack([Xnzi, Xenso])

    if n is not None:
        X = X[:n]

    # NZI/ENSO etc
    Xnzi = X[:, fts[fts.nzi>0].index].mean(axis=1).reshape(-1,1)
    Xenso = X[:, fts[fts.enso==3.4].index].mean(axis=1).reshape(-1,1)
    Xtele = np.hstack([Xnzi, Xenso])

    return X, Xlens, fts, Xnzi, Xenso, Xtele

def load_precipitation(all_divs=False):
    df = pd.DataFrame()
    for a, b, files in os.walk('data/new_precip/'):
        for f in files:
            if 'prec' in f:
                tmp = pd.read_csv('data/new_precip/' + f,
                                    header= None,
                                    names= ['date', 'precip', 'diff_from_mean'])
                tmp['file'] = '_'.join(f.split('_')[1:3])
                df = df.append(tmp)
    # we only keep period 03/1941- 03/2015
    df = df[(df.date>=194103)]
    areas = pd.read_csv('data/new_precip/ClDiv_areas.txt', header=None, sep='.')
    areas = areas.drop(11, axis=1)
    areas['st'] = areas[0].apply(lambda x: x.split()[0])
    areas[0] = areas[0].apply(lambda x: x.split()[1])
    cols = 'DIV1   DIV2   DIV3   DIV4   DIV5   DIV6   DIV7   DIV8   DIV9  DIV10 STATE_AREA ST'.lower().split()
    areas.columns = cols
    # according to https://www.esrl.noaa.gov/psd/data/usclimdivs/descript.html, we have the following state mappings
    # california(04), arizona(02), nevada(26), utah(42)
    areas = areas[areas.st.isin(['04', '02', '26', '42'])]
    areas['state'] = ['Arizona', 'California', 'Nevada', 'Utah']
    areas = pd.melt(areas, id_vars=['state'], value_vars=['div1', 'div2', 'div3', 'div4', 'div5', 'div6', 'div7'])
    areas.columns=['state', 'division', 'area']
    areas['area'] = areas.area.astype(int)
    areas['division'] = areas.division.apply(lambda x: int(x.replace('div', '')))
    areas['file'] = areas.apply(lambda x: str(x.division)+"_"+x.state, axis=1)
    # merge data
    precip = pd.merge(df, areas[['file', 'area']], on='file')
    precip['year'] = precip.date.apply(lambda x: int(np.floor(x/100)))
    # these are the divisions that we consider for the mean precipitation amount
    # see Figure 1 and discussion in Mamalakis et al (2018) for more info
    divs2keep = [
        '4_California',
        '5_California',
        '6_California',
        '7_California',
        '1_Arizona',
        '2_Arizona',
        '3_Arizona',
        '4_Arizona',
        '5_Arizona',
        '6_Arizona',
        '7_Arizona',
        '3_Nevada',
        '4_Nevada',
        '1_Utah',
        '2_Utah',
        '4_Utah',
        '6_Utah',
        '7_Utah'
    ]
    if not all_divs:
        return precip[precip.file.isin(divs2keep)]
    else:
        return precip

def load_response(n):
    precip = load_precipitation()
    Y = precip.pivot('year', 'file', 'precip').values
    Y = preprocessing.scale(Y)
    Y = Y[:n]

    precip['weighted_precip'] = precip.apply(lambda x: x.precip*x.area, axis=1)
    avg_rain = (precip.groupby('date').weighted_precip.sum()/sum(precip.area.unique())).reset_index()
    y = np.array(avg_rain.weighted_precip)
    yavg = preprocessing.scale(y)
    yavg = yavg[:n]

    regions = [x.split( '_')[1]+'({})'.format(x.split('_')[0]) for x in precip.pivot('year', 'file', 'precip').columns]

    return yavg, Y, regions


def televaluation(Xnzi, Xenso, Xtele, y, region, alpha, t):
    tele_errors = []
    lm = linear_model.LinearRegression()

    if alpha == 1:
        X_train, X_test, y_train, y_test = temporal_split(Xnzi, y, t)
        lm.fit(X_train, y_train)
        tele_errors.append(['NZI', region,
                            compute_errors(y_test, X_test@lm.coef_)[0],
                            compute_errors(y_test, X_test@lm.coef_)[1],
                           lm.coef_])

        X_train, X_test, y_train, y_test = temporal_split(Xenso, y, t)
        lm.fit(X_train, y_train)
        tele_errors.append(['Nino 3.4', region,
                            compute_errors(y_test, X_test@lm.coef_)[0],
                            compute_errors(y_test, X_test@lm.coef_)[1],
                           lm.coef_])

        X_train, X_test, y_train, y_test = temporal_split(Xtele, y, t)
        lm.fit(X_train, y_train)
        tele_errors.append(['Nino 3.4 & NZI', region,
                            compute_errors(y_test, X_test@lm.coef_)[0],
                            compute_errors(y_test, X_test@lm.coef_)[1],
                           lm.coef_])
    else:
        n = t
        weights = np.array([alpha**(n-t) for t in np.arange(1, n+1)])
        X_train, X_test, y_train, y_test = temporal_split(Xnzi, y, t)
        lm.fit(X_train, y_train, sample_weight=weights)
        tele_errors.append(['NZI', region,
                            compute_errors(y_test, X_test@lm.coef_)[0],
                            compute_errors(y_test, X_test@lm.coef_)[1],
                           lm.coef_])

        X_train, X_test, y_train, y_test = temporal_split(Xenso, y, t)
        lm.fit(X_train, y_train, sample_weight=weights)
        tele_errors.append(['Nino 3.4',  region,
                            compute_errors(y_test, X_test@lm.coef_)[0],
                            compute_errors(y_test, X_test@lm.coef_)[1],
                           lm.coef_])

        X_train, X_test, y_train, y_test = temporal_split(Xtele, y, t)
        lm.fit(X_train, y_train, sample_weight=weights)
        tele_errors.append(['Nino 3.4 & NZI', region,
                            compute_errors(y_test, X_test@lm.coef_)[0],
                            compute_errors(y_test, X_test@lm.coef_)[1],
                           lm.coef_])
    df_tele = pd.DataFrame(tele_errors, columns=['Method', 'Region', 'MSE', 'R2', 'Coefs'])
    df_tele['alpha'] = alpha
    return df_tele
