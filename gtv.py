import numpy as np
import pandas as pd

# glmnet now
import glmnet_python
import scipy, importlib, pprint, matplotlib.pyplot as plt, warnings
from glmnet import glmnet;
from glmnetPlot import glmnetPlot
from glmnetPrint import glmnetPrint; from glmnetCoef import glmnetCoef; from glmnetPredict import glmnetPredict
from cvglmnet import cvglmnet; from cvglmnetCoef import cvglmnetCoef
from cvglmnetPlot import cvglmnetPlot; from cvglmnetPredict import cvglmnetPredict
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from utils import compute_errors, temporal_split

def edge_incidence(S, threshold=0):
    edges = np.where(abs(S)>threshold)
    edges_ix = np.where(edges[0]<edges[1])
    ix1 = edges[0][edges_ix]
    ix2 = edges[1][edges_ix]
    ix = np.arange(len(ix1))

    D = np.zeros([len(ix1), S.shape[0]])
    D[ix, ix1] = np.sqrt(abs(S[ix1, ix2]))
    D[ix, ix2] = -np.sign(S[ix1, ix2])*np.sqrt(abs(S[ix1, ix2]))
    return D

# build augmented system
def augmented_system_lasso(X,y,D,lam1,lam2, l1_only=False):
    [n,p] = X.shape
    [m,p] = D.shape
    if l1_only:
        bigX = X
        bigY = y
    else:
        bigX = np.vstack([X, np.sqrt(n*lam2)*D])
        bigY = np.append(y, np.zeros(m))
    bigD = np.vstack([lam1*D, np.identity(p)])
    invD = np.linalg.solve(bigD.T @ bigD, bigD.T)
    XD = bigX @ invD
    return XD, bigY, invD

def build_pairs(df):
    # this is inefficient but whatever
    time_pairs = []
    # find gap in locations (they'll be the same for lat/lon by design)
    locs = sorted(df.lat.unique())
    loc_jump = locs[1] - locs[0]
    for i in range(1, max(df.time) + 1):
        ix1 = df.index[df.time == i].tolist()
        ix0 = df.index[df.time == i - 1].tolist()
        time_pairs = time_pairs + list(zip(ix1, ix0))

    loc_pairs = []
    for i in range(min(df.lat), max(df.lat) + 1, loc_jump):
        for j in range(min(df.lon), max(df.lon) + 1, loc_jump):
            for k in df.time.unique():
                ix1 = df.index[(df.lat==i)&(df.lon==j)&(df.time==k)].tolist()
                ix0 = df.index[(df.lat.isin([i, i+loc_jump, i-loc_jump])&(df.time==k))
                               & (df.lon.isin([j, j+loc_jump, j-loc_jump]))&(df.time==k)].tolist()
                loc_pairs = loc_pairs + list(zip(ix1*len(ix0),ix0))
    df = pd.DataFrame(time_pairs+loc_pairs, columns=['ix1', 'ix2'])
    df['pair_type'] = 'location'
    if len(time_pairs) > 0:
        df.loc[:len(time_pairs), 'pair_type'] = 'time'
    df = df[df.ix1 != df.ix2]
    # order them so smallest is first
    pairs = df
    swap_index = pairs[pairs.ix1 > pairs.ix2][['ix1', 'ix2']].index
    ix1 = pairs.loc[swap_index, 'ix1']
    ix2 = pairs.loc[swap_index, 'ix2']
    pairs.loc[swap_index, 'ix1'] = ix2
    pairs.loc[swap_index, 'ix2']= ix1
    pairs = pairs.drop_duplicates(['ix1', 'ix2'])
    pairs = pairs.reset_index().drop('index', axis=1)
    return pairs

def neighbors(df, W=None):
    pairs = pd.DataFrame()
    for a, grps in df[df.variable.isin(['sst', 'gph'])].groupby('variable'):
        pairs = pairs.append(build_pairs(grps), ignore_index=True)
    dupes = pairs.merge(pairs.reset_index(), left_on=['ix1', 'ix2'], right_on=['ix2', 'ix1'])['index']
    pairs = pairs.loc[~pairs.index.isin(dupes)]
    pairs = pairs.reset_index()
    if W is not None:
        pairs['corr_W'] = pairs.apply(lambda x: W[x.ix1, x.ix2], axis=1)
        D = np.zeros([pairs.shape[0], df.shape[0]])
        D[list(pairs.index), list(pairs.ix2)] = np.sqrt(np.abs(pairs.corr_W))
        D[list(pairs.index), list(pairs.ix1)] = -np.sign(pairs.corr_W)*np.sqrt(np.abs(pairs.corr_W))
    else:
        D = np.zeros([pairs.shape[0], df.shape[0]])
        D[list(pairs.index), list(pairs.ix2)] = 1
        D[list(pairs.index), list(pairs.ix1)] = -1
    return D

def weighted_gtv(X, y, D, l1, l3, alpha=.9):
    if alpha<1:
        n = X.shape[0]
        weights = np.array([alpha**(n-t) for t in np.arange(1, n+1)])
        X = X * np.sqrt(weights.reshape(-1,1))
        y = y * np.sqrt(weights)
    XD, bigY, invD = augmented_system_lasso(X, y, D, l1/l3, 0, l1_only=True)
    fit = glmnet(x = XD, y = bigY)
    b = glmnetCoef(fit, s = scipy.float64([l3]), exact = False)
    beta = invD@b.reshape(b.shape[0])[1:]
    return beta


def full_weighted_cv(X, y, Ds, lambda_gtv=np.linspace(.1, 1, 10), lambda_lasso=None, t=50, auto_cv=True, alpha=.9, k=5):
    errors = []
    X_train, X_test, y_train, y_test = temporal_split(X, y, t)
    if alpha<1:
        n = X_train.shape[0]
        weights = np.array([alpha**(n-t) for t in np.arange(1, n+1)])
        X_train = X_train * np.sqrt(weights.reshape(-1,1))
        y_train = y_train * np.sqrt(weights)
    n,p = X_train.shape
    # test errors
    for l1 in lambda_gtv:
        for m in Ds:
            D = Ds[m]
            if auto_cv:
                XD, bigY, invD = augmented_system_lasso(X_train, y_train, D, l1, 0, l1_only=True)
                fit = cvglmnet(x = XD, y = bigY, family = 'gaussian', ptype = 'mse', nfolds = 5)
                b = cvglmnetCoef(fit, s = 'lambda_min')
                l3 = fit['lambda_min'][0]
                beta = invD@b.reshape(b.shape[0])[1:]
                mset, r2t = compute_errors(y_train, X_train@beta)
                mse, r2 = compute_errors(y_test, X_test@beta)
                errors.append([m, l1, l3, mset, r2t, mse, r2])
            else:
                for l3 in lambda_lasso:
                    XD, bigY, invD = augmented_system_lasso(X_train, y_train, D, l1/l3, 0, l1_only=True)
                    #XD, bigY, invD = epsilon_system_lasso(X_train, y_train, D, l1)
                    fit = glmnet(x = XD, y = bigY)
                    b = glmnetCoef(fit, s = scipy.float64([l3]), exact = False)
                    beta = invD@b.reshape(b.shape[0])[1:]
                    mset, r2t = compute_errors(y_train, X_train@beta)
                    mse, r2 = compute_errors(y_test, X_test@beta)
                    errors.append([m, l1, l3, mset, r2t, mse, r2])
    df = pd.DataFrame(errors, columns=['method', 'lambda_tv', 'lambda_1', 'train_mse', 'train_r2', 'test_mse', 'test_r2'])
    return df
