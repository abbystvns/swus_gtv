import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import seaborn as sns


legend_elements = [Line2D([0], [0], marker='o', color='silver', label='Positive coefficient',
                          markerfacecolor='yellow',  markersize=15),
                   Line2D([0], [0], marker='o', color='silver', label='Negative coefficient',
                          markerfacecolor='rebeccapurple',  markersize=15)]

def plot_coefs(df, title='', shade_time=False, filename=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6.5)

    m = Basemap(projection='merc', \
                llcrnrlat=-60, urcrnrlat=60, \
                llcrnrlon=80, urcrnrlon=280, \
                lat_ts=20, \
                resolution='c')

    m.shadedrelief(scale=.2)
    m.drawcoastlines(color='black', linewidth=0.5)  # add coastlines
    m.drawparallels(np.arange(-60,70,10), labels=[1,0,0,0], linewidth=.1)
    meridians = m.drawmeridians(np.arange(80,281,10), labels=[0,0,0,1], linewidth=.1)
    for k in meridians:
        try:
            meridians[k][1][0].set_rotation(45)
        except:
            pass


    #nzi
    ur = m(200, -25)
    lr = m(170, -40)
    boxes = [Rectangle(lr, ur[0]-lr[0], ur[1]-lr[1])]
    #boxes = []

    #nino
    for pts in [(190, 240)]:
        ur = m(pts[1], 5)
        lr = m(pts[0], -5)
        box = Rectangle(lr, ur[0]-lr[0], ur[1]-lr[1])
        boxes.append(box)

    pc = PatchCollection(boxes, edgecolor='blue', alpha=.1)
    ax.add_collection(pc)

    if shade_time:
        alph = 1
        for a, grp in df.groupby('time'):
            lats = list(grp.lat)
            lons = list(grp.lon)
            coefs = np.array(abs(grp.coef))
            sign = np.sign(grp.coef)
            x, y = m(lons, lats)  # transform coordinates
            plt.scatter(x, y, s=np.sqrt(coefs)*1000, c=sign, label=sign, alpha=alph)
            alph-=.2
    else:
        lats = list(df.lat)
        lons = list(df.lon)
        coefs = np.array(abs(df.coef))
        sign = np.sign(df.coef)
        x, y = m(lons, lats)  # transform coordinates
        plt.scatter(x, y, s=coefs*1000, c=sign, label=sign)

    #ax.legend(handles=legend_elements, loc='upper left', facecolor='silver')
    plt.title(title)
    if filename is not None:
        plt.savefig(filename)

def plot_coefs_by_time(df, save=False):
    nonz = df[abs(df.coef)>0]
    for b, big_grp in nonz.groupby('variable'):
        var = big_grp.variable.unique()[0]
        for a, grp in big_grp.groupby('month'):
            time = grp.month.unique()[0]
            if save:
                plot_coefs(grp, str(time), filename='images/'+var+str(time))
            else:
                plot_coefs(grp, str(time))


def plot_covariance(covs, titles, h = 4, ori='vertical', bar=True):
    n = len(covs)
    if bar:
        fig = plt.figure(figsize=(n*h+h/2, h))
    else:
        fig = plt.figure(figsize=(n*h, h))
    axes=[]
    plots = []
    subplot_ix = 100 + n*10
    for i in range(n):
        ax = fig.add_subplot(subplot_ix+i+1)
        axes.append(ax)
    for i, S in enumerate(covs):
        g = sns.heatmap(S, ax=axes[i], vmin=-1, vmax=1, cmap='coolwarm', xticklabels=False, yticklabels=False, cbar=False)
        plots.append(g)
        plots[i].set_title(titles[i])
    mappable = plots[0].get_children()[0]
    if bar:
        plt.colorbar(mappable, ax = axes, orientation = ori)


def plot_heatmap(fts, s, month, title=None, scale_coefs=100, vmin=-.3, vmax=.3):
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6.5)
    m = Basemap(projection='merc', \
                llcrnrlat=-60, urcrnrlat=60, \
                llcrnrlon=80, urcrnrlon=280, \
                lat_ts=20, resolution='c')

    #m.shadedrelief(scale=.2)
    m.drawcoastlines(color='black', linewidth=0.5)  # add coastlines
    m.drawparallels(np.arange(-60,60,20), labels=[1,0,0,0], linewidth=.01)
    m.drawmeridians(np.arange(80,280,20), labels=[0,0,0,1], linewidth=.01)

    lats = list(fts[fts.month==month].lat)
    lons = list(fts[fts.month==month].lon)
    coef_matrix = fts[fts.month==month].pivot_table(index='lat', columns='lon', values='coef').values
    coef_matrix = np.flip(coef_matrix, axis=0)
    coef_matrix = np.nan_to_num(coef_matrix, 0)
    resized_coefs = skimage.transform.resize(coef_matrix, (coef_matrix.shape[0] * s, coef_matrix.shape[1] * s),
                           anti_aliasing=False, order=1)

    resized_coefs = skimage.filters.gaussian(resized_coefs, .1)

    new_lats = list(np.linspace(fts.lat.min(), fts.lat.max(), fts.lat.nunique()*s))*fts.lon.nunique()*s
    new_lats.sort(reverse=True)
    new_lons = list(np.linspace(fts.lon.min(), fts.lon.max(), fts.lon.nunique()*s))*fts.lat.nunique()*s

    coefs = resized_coefs.reshape((len(new_lats),))

    x, y = m(new_lons, new_lats)  # transform coordinates
    #p = plt.scatter(x, y, s=np.sign(abs(coefs))*10,c=coefs*100, cmap='bwr', vmin=-.3, vmax=.3)
    p = plt.scatter(x, y, s=np.sign(abs(coefs))*10, c=coefs*scale_coefs, cmap='bwr', vmin=vmin, vmax=vmax)
    plt.title(title)
    m.colorbar(p)
    return fig

def draw_lambda_contour(df, method, variable, vmin=None, vmax=None):
    if 'mse' in df.columns:
        df = df.melt(id_vars=['method', 'lambda_tv', 'lambda_1'])
    #df[['lambda_1', 'lambda_tv']] = round(df[['lambda_1', 'lambda_tv']], 2)
    d = df[(df.method==method)&(df.variable==variable)].pivot(index='lambda_1', columns='lambda_tv', values='value')
    if vmin is None:
        vmin = df[(df.variable==variable)].value.min()
        vmax = df[(df.variable==variable)].value.max()
    d = d.sort_index(ascending=False)
    sns.heatmap(d, vmin=vmin, vmax=vmax, cmap='coolwarm')
    plt.xlabel('$\lambda_{TV}$', fontweight='bold')
    plt.ylabel('$\lambda_1$', fontweight='bold')
    plt.xticks(rotation='vertical')
    #plt.xticks(np.arange(0, 10, step=1))
    plt.yticks(rotation='horizontal')


def swap_region(region):
    if 'Average' in region:
        return region
    elif len(region) > 5:
        state_map = {
            'Arizona':'AZ',
            'Utah':'UT',
            'Nevada':'NV',
            'California':'CA'
        }
    else:
        state_map = {
            'AZ':'Arizona',
            'UT':'Utah',
            'NV':'Nevada',
            'CA':'California',
        }
    return state_map[region.split('(')[0]] + '(' + region.split('(')[1]


def shorten_region(region):
    if region in ['Area-weighted average', 'Areal Average']:
        return 'Areal \n Average'
    if len(region) > 5:
        state_map = {
            'Arizona': 'AZ',
            'Utah': 'UT',
            'Nevada': 'NV',
            'California': 'CA'
        }
        return state_map[region.split('(')[0]] + '(' + region.split('(')[1]
    else:
        return region


def plot_regional_metrics(df, metric='mse', figsize=(25,5)):
    """
    Dataframe should have the following columns:
    - Region
    - Method
    - Test MSE or R2
    """
    # make sure regions are short
    df['Region'] = df.Region.apply(lambda x: shorten_region(x))
    plt.figure(figsize=figsize)
    sns.set(font_scale=2)
    if metric == 'mse':
        sns.barplot(x='Region', y='Test MSE', hue='Method', data=df)
        plt.ylabel('MSE', fontweight='bold')
    elif metric == 'r2':
        df['R2'] = df.R2.apply(lambda x: max(.01, x))
        sns.barplot(x='Region', y='R2', hue='Method', data=df)
        plt.ylabel('$R^2$', fontweight='bold')
    plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                    mode="expand", borderaxespad=0, ncol=4)
    plt.xlabel('Region', fontweight='bold')
