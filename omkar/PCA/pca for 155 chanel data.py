#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
#get_ipython().magic(u'matplotlib inline')


# In[4]:


x = pd.read_csv('/Users/ombahiwal/PycharmProjects/test_kvb/omkar/CNN/train3/x_all.csv',delimiter=",")
y = pd.read_csv('/Users/ombahiwal/PycharmProjects/test_kvb/omkar/CNN/train3/y_all.csv',delimiter=",")


# In[5]:


np.shape(x)


# In[6]:


np.shape(y)


# In[7]:


x


# In[8]:


y


# In[9]:


X = x.iloc[:,1:156].values
y = y.iloc[:,0].values
X


# In[10]:


y


# In[11]:


np.shape(x)


# In[12]:


np.shape(y)


# In[13]:


from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)


# In[14]:


mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)


# In[15]:


print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))


# In[16]:


plt.figure(figsize=(16,16))
sns.heatmap(cov_mat, vmax=1, square=True,annot=True,cmap='cubehelix')

plt.title('Correlation between different features')
# In[17]:


eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# In[18]:


# 6) Selecting Principal Components¶
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])


# In[19]:


tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
var_exp


# matrix_w = np.hstack((eig_pairs[0][1].reshape(7,1), 
#                       eig_pairs[1][1].reshape(7,1)
#                     ))
# print('Matrix W:\n', matrix_w)

# In[20]:


matrix_w = np.hstack((eig_pairs[0][1].reshape(154,1), 
                      eig_pairs[1][1].reshape(154,1)
                    ))
print('Matrix W:\n', matrix_w)


# In[30]:


Y = X_std.dot(matrix_w)
Y


# In[33]:
#clear previous plot
plt.gcf().clear()
from sklearn.decomposition import PCA
pca = PCA().fit(X_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0, 155, 1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')


# In[39]:


print (np.cumsum(pca.explained_variance_ratio_))
import xlsxwriter

workbook = xlsxwriter.Workbook('cumm.xlsx')
worksheet = workbook.add_worksheet()



# In[ ]:





# In[42]:


from sklearn.decomposition import PCA

# Take this value as input 40 n_components
sklearn_pca = PCA(n_components=40)
Y_sklearn = sklearn_pca.fit_transform(X_std)


# In[43]:


print(Y_sklearn)


# In[44]:


Y_sklearn.shape


# In[45]:


import xlsxwriter

# Reduced Generated file, Get file name from the user
workbook = xlsxwriter.Workbook('arrays.xlsx')
worksheet = workbook.add_worksheet()


# In[46]:


row = 0

for col, data in enumerate(Y_sklearn):
    worksheet.write_column(row, col, data)

workbook.close()




"""
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkMessageBox

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
matplotlib.use('TkAgg')
import numpy as np

from Tkinter import *
import Tkinter as tk
import tkFileDialog
class mclass:

    def __init__(self,  window):
        self.window = window
        self.box = Entry(window)
        self.box.insert(0, 'enter filename')
        self.button = Button(window, text="Apply PCA to Datasets", command=self.plot)
        self.button2 = Button(window, text="Generate File", command=self.generate_file)
        self.comp = Entry(window)
        self.comp.insert(0, 'enter no. of components')
        self.button.pack()

        self.plot_values = []

    def get_filepath(self):
        fpath = tkFileDialog.askopenfilename()
        return str(fpath)

    def plot(self):
        tkMessageBox.showinfo("Message", "Select X test file")

        x = pd.read_csv(self.get_filepath(), delimiter=",")
        tkMessageBox.showinfo("Message", "Select Y train file")
        y = pd.read_csv(self.get_filepath(), delimiter=",")
        tkMessageBox.showinfo("Processing", "Please wait...\n")
        np.shape(x)
        np.shape(y)
        X = x.iloc[:, 1:156].values
        y = y.iloc[:, 0].values
        np.shape(x)
        np.shape(y)
        from sklearn.preprocessing import StandardScaler
        self.X_std = StandardScaler().fit_transform(X)
        mean_vec = np.mean(self.X_std, axis=0)
        cov_mat = (self.X_std - mean_vec).T.dot((self.X_std - mean_vec)) / (self.X_std.shape[0] - 1)
        print('Covariance matrix \n%s' % cov_mat)
        print('NumPy covariance matrix: \n%s' % np.cov(self.X_std.T))
        plt.figure(figsize=(16, 16))
        sns.heatmap(cov_mat, vmax=1, square=True, annot=True, cmap='cubehelix')
        plt.title('Correlation between different features')
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)

        for i in eig_pairs:
            print(i[0])
        tot = sum(eig_vals)
        var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
        matrix_w = np.hstack((eig_pairs[0][1].reshape(154, 1),
                              eig_pairs[1][1].reshape(154, 1)
                              ))
        print('Matrix W:\n', matrix_w)
        Y = self.X_std.dot(matrix_w)
        plt.gcf().clear()
        from sklearn.decomposition import PCA
        pca = PCA().fit(self.X_std)
        self.plot_values = np.cumsum(pca.explained_variance_ratio_)
        plt.plot(self.plot_values)
        plt.xlim(0, 155, 1)
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative explained variance')
        fig = Figure(figsize=(6, 6))
        a = fig.add_subplot(111)
        a.plot(self.plot_values)
        a.set_title("PCA Variance Ratio", fontsize=16)
        a.set_ylabel('Cumulative explained variance', fontsize=14)
        a.set_xlabel('Number of components', fontsize=14)

        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.get_tk_widget().pack()
        canvas.draw()
        self.comp.pack()
        self.box.pack()
        self.button2.pack()
        tkMessageBox.showinfo("PLOT", "Graph Generated")
        print (self.plot_values)

    def generate_file(self, ):
        import xlsxwriter
        components = int(self.comp.get())
        fname = str(self.box.get())
        from sklearn.decomposition import PCA
        # Take this value as input 40 n_components
        self.sklearn_pca = PCA(n_components=components)
        self.Y_sklearn = self.sklearn_pca.fit_transform(self.X_std)
        print(self.Y_sklearn)
        self.Y_sklearn.shape
        import xlsxwriter
        # Reduced Generated file, Get file name from the user
        path = '/Users/ombahiwal/Documents/Interface_Tool/'+fname+'.xlsx'
        workbook = xlsxwriter.Workbook(path)
        worksheet = workbook.add_worksheet()
        row = 0
        for col, data in enumerate(self.Y_sklearn):
            worksheet.write_column(row, col, data)
        workbook.close()
        tkMessageBox.showinfo("Message", "Dataset File with "+str(components)+" components generated at "+path)

window = Tk()
window.geometry('600x1000')
title = tk.Label(window, text="Interface Tool for REP Analysis",  padx=10)
title.config(font=("Helvetica", 25))
title.pack()

start = mclass(window)

window.mainloop()

"""