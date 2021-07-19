import numpy as np 
import glob
from sklearn.manifold import TSNE,Isomap
import os
import matplotlib.pyplot as plt
from scipy.linalg import null_space


def tsne_plot(featureFile,n_components=2, random_state=42):
    """
    Create a TSNE model and plot it
    """
    print("-- Start t-SNE plot --")
    labels = []
    tokens = []
    
    for tag,embedding in featureFile:
        
        tokens.append(np.squeeze(embedding,0))
        labels.append(tag)

    tsne_model = TSNE(n_components=n_components, random_state=random_state)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    
    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(5, 2), textcoords="offset points", ha="right", va="bottom")
    

def isomap(featureFile,n_components=3, random_state=42):
    """
    Create a TSNE model and plot it
    """
    print("-- Start t-SNE plot --")
    labels = []
    tokens = []
    
    for tag,embedding in featureFile:
        
        tokens.append(np.squeeze(embedding,0))
        labels.append(tag)

    isomap = Isomap(n_components=n_components)
    new_values = isomap.fit_transform(tokens)

    x = []
    y = []
    z= []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    
    
    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(5, 2), textcoords="offset points", ha="right", va="bottom")
    


def loadData():
    ### loading the numpy file for testing
    
    filePath = "/home/gopal/Desktop/shubhamCode/shubham/baselines/images/continuos/"
    fileName = os.path.join(filePath, "trajectory.npy")
    data = np.load(fileName, allow_pickle = True)
    dataList = []
    data = dict(np.ndenumerate(data))

    # dataList.append(("hyperplane", data[()]['hyperplane'].detach().cpu().numpy()))
    dataList.append(("base_state", data[()]['base_state']))
    hyperplane_normals = data[()]['hyperplane'].detach().cpu().numpy()
    hyperplane_null_space_basis = null_space(hyperplane_normals)

    for i, step in enumerate(data[()]['step_list']):
        # print(step.shape)
        dataList.append( ("s"+str(i+1), step) ) 

    for i, basis in enumerate(hyperplane_null_space_basis.T):
        # print(basis.shape)
        dataList.append(('b', np.expand_dims(basis, 0))) 


    tsne_plot(featureFile=dataList)
    # isomap(featureFile=dataList)
    plt.show()

if __name__ =="__main__":
    loadData()
