import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import plotly.express as px
from sklearn.decomposition import PCA
from import_HD_data import * 
from PCA_3 import V

# Creating figure
#fig = plt.figure(figsize = (10, 7))
#ax = plt.axes(projection ="3d")
 
# Creating plot
o = np.array([[0,0],[0,0]])
plt.quiver(*o,V[:,0],V[:,1],color=['blue','green'])
plt.title("simple 3D scatter plot")
 
# show plot
plt.show()

# features = ["SBP","CT","LDL","BAI","FAM","TA","BMI","CAC","AGE"]
# pca = PCA()
# components = pca.fit_transform(X[features])
# labels = {
#     str(i): f"PC {i+1} ({var:.1f})"
#     for i, var in enumerate(pca.explained_variance_ratio_*100)
# }
# fig = px.scatter_matrix(
#     components,
#     labels=labels,
#     dimensions=range(10)
# )
# fig.update_traces(diagonal_visible=False)
# fig.show()

#X1 = df1[['sbp','tobacco','ldl','adiposity','famhist','typea','obesity','alcohol','age']]
#pca = PCA(n_components=3)
#components = np.array([V[:,0],V[:,1],V[:,2]])
#components1 = pca.fit_transform(X)
#print('Comp:',components)
#print(components1)
#fig = px.scatter_3d(components1,x=0,y=0,z=0,title='Components',labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'})
#fig.show()