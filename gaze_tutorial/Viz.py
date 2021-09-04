# Library of functions for plotting gaze data from multidimensional
# reinforcement learning task. 

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as MT
import matplotlib.lines as L
import matplotlib.cm as CM
import matplotlib.colors as C
import matplotlib.patches as PA

def plot_attention_pdf(world, data, sim):
    """ Plots attention timecourses in the FHT task.
        Style: timecourse of probability distributions. 

        Parameters
        ----------
        world: instance of World.

        data: float, array(n_trials, )
            Can have dimensionality n_feats or n_dims, depending on whether 
            we plot feature level or dimension level attention.

        sim: boolean
            Flags whether this is real (0) or simulated data (1). 
    """

    n_trials, d = data.shape

    ## Define colormap
    # colors = ['#1f78b4', '#33a02c', '#ff7f00']
    colors = ['#0571b0', '#ca0020', '#404040']

    fig, ax = plt.subplots(1, 1, figsize=(30,6));
    # fig, ax = plt.subplots(1, 1, figsize=(1,6));
    ax.set_xlim((0, n_trials+1));
    ax.set_xticks(np.arange(n_trials)+1);
    ax.set_yticks(np.arange(d)+1);
    xl = ax.set_xlabel('Trial',fontsize = 30);
    if not sim: yl = ax.set_ylabel('Feature',fontsize = 30); 
    else: yl = ax.set_ylabel('Feature',fontsize = 30); 
    ax.tick_params(labelsize=30)
    plt.ylim([0,d+1])
    # sns.despine()

    for f in np.arange(d):

        feat = f+1
        x = np.arange(n_trials)+1
        y = feat * np.ones(n_trials)
        s = data[feat].values*1000
        if d == 3:
            plt.scatter(x,y,s=s,color=colors[f],marker='s')
        else:
            if 1 <= feat <= 3: plt.scatter(x,y,s=s,color=colors[0],marker='s')
            if 4 <= feat <= 6: plt.scatter(x,y,s=s,color=colors[1],marker='s')
            if 7 <= feat <= 9: plt.scatter(x,y,s=s,color=colors[2],marker='s')
            
    return fig, ax

def plot_attention_simplex(world, data):

    """ Plots attention timecourses in the FHT task.
        Style: simplex projection. 

        Parameters
        ----------
        world: instance of World.

        data: float, array(n_trials, )
            Can have dimensionality n_feats or n_dims, depending on whether 
            we plot feature level or dimension level attention.

    """

    # Define different colors for each label
    cmap = CM.get_cmap('Reds')
    norm = C.Normalize(vmin=0, vmax=len(data))
    c = range(len(data))
    # Do scatter plot
    fig = plotSimplex(data, s=400, c=c, cmap=cmap, norm=norm, linewidth=2)

    return fig

def plot_attention_line(world, data):

    """ Plots attention timecourses in the FHT task.
        Style: line plot. 

        Parameters
        ----------
        world: instance of World.

        data: DataFrame(n_trials, )
            Can have n_feats or n_dims columns, depending on whether 
            we plot feature level or dimension level attention.

    """

    trials = np.arange(len(data))

    ## Make plot.
    fig, ax = plt.subplots(figsize=(20,8), linewidth=6);
    ax = sns.lineplot(data=data, palette=['#1f78b4', '#33a02c', '#ff7f00'], lw=6, legend=False, linestyle='-')
    ax.lines[0].set_linestyle("-")
    ax.lines[1].set_linestyle("-")
    ax.lines[2].set_linestyle("-")
    ax.set_xticks(trials)
    ax.set_xticklabels(trials+1)
    ax.set_yticks([0,0.5,1])
    ax.tick_params(labelsize=30)
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(3)
    plt.ylim([0,1.05])
    sns.despine()

    return fig

def plotSimplex(points, fig=None, 
                vertexlabels=['Faces','Houses','Tools'],
                **kwargs):
    """
    Plot Nx3 points array on the 3-simplex 
    (with optionally labeled vertices) 
    
    kwargs will be passed along directly to matplotlib.pyplot.scatter    

    Returns Figure, caller must .show()
    """
    if(fig == None):        
        fig = plt.figure(figsize=(10,10))
    # Draw the triangle
    l1 = L.Line2D([0, 0.5, 1.0, 0], # xcoords
                  [0, np.sqrt(3) / 2, 0, 0], # ycoords
                  color='k')
    fig.gca().add_line(l1)
    fig.gca().xaxis.set_major_locator(MT.NullLocator())
    fig.gca().yaxis.set_major_locator(MT.NullLocator())
    # Draw vertex labels
    # fig.gca().text(-.25, -0.05, vertexlabels[0], size=30)
    # fig.gca().text(1.05, -0.05, vertexlabels[1], size=30)
    # fig.gca().text(0.5-0.1, np.sqrt(3) / 2 + 0.05, vertexlabels[2], size=30)
    # Project and draw the actual points
    projected = projectSimplex(points)
    plt.scatter(projected[:,0], projected[:,1], **kwargs)
    # Annotate with time point 
    tp = np.arange(len(points))+1
    for i in np.arange(len(points)):
        fig.gca().annotate(tp[i], (projected[i,0]-0.01, projected[i,1]-0.017),size=12)              
    # Leave some buffer around the triangle for vertex labels
    fig.gca().set_xlim(-0.2, 1.2)
    fig.gca().set_ylim(-0.2, 1.2)
    fig.gca().axis('off')

    return fig    

def projectSimplex(points):
    """ 
    Project probabilities on the 3-simplex to a 2D triangle
    
    N points are given as N x 3 array
    """
    # Convert points one at a time
    tripts = np.zeros((points.shape[0],2))
    for idx in range(points.shape[0]):
        # Init to triangle centroid
        x = 1.0 / 2
        y = 1.0 / (2 * np.sqrt(3))
        # Vector 1 - bisect out of lower left vertex 
        p1 = points[idx, 0]
        x = x - (1.0 / np.sqrt(3)) * p1 * np.cos(np.pi / 6)
        y = y - (1.0 / np.sqrt(3)) * p1 * np.sin(np.pi / 6)
        # Vector 2 - bisect out of lower right vertex  
        p2 = points[idx, 1]  
        x = x + (1.0 / np.sqrt(3)) * p2 * np.cos(np.pi / 6)
        y = y - (1.0 / np.sqrt(3)) * p2 * np.sin(np.pi / 6)        
        # Vector 3 - bisect out of top vertex
        p3 = points[idx, 2]
        y = y + (1.0 / np.sqrt(3) * p3)
      
        tripts[idx,:] = (x,y)

    return tripts
