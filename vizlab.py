import sklearn
import pandas as pd
from sklearn.metrics import silhouette_samples,silhouette_score
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import io
from fpdf import FPDF
from pymsgbox import *
sns.set_style('darkgrid')
class visualize:
    def viz_clf(clfs,X,y,cmap='jet',al=0.5,figure_size=(12,8)):
        if type(clfs)!=list:
            if X.shape[1]!=2:
                raise TypeError('X should contain only 2 features')
            plt.figure(figsize=figure_size)
            plt.title("The visualization of the classification data.")
            plt.xlabel("Feature space for the 1st feature")
            plt.ylabel("Feature space for the 2nd feature")
            plt.suptitle(clfs.__class__.__name__,
                    fontsize=14, fontweight='bold')
            clfs.fit(X,y)
            x1,x2=X[:,0].min()-1,X[:,0].max()+1
            y1,y2=X[:,1].min()-1,X[:,1].max()+1
            xx,xy=np.meshgrid(np.arange(x1,x2,0.1),np.arange(y1,y2,0.1))
            z=clfs.predict(np.c_[xx.ravel(),xy.ravel()])
            z=z.reshape(xx.shape)
            plt.contourf(xx, xy, z, cmap=cmap, alpha=al)
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap)
            plt.show()
        else:
            for i,clf in enumerate(clfs):
                plt.figure(figsize=figure_size)
                plt.title("The visualization of the classification data.")
                plt.xlabel("Feature space for the 1st feature")
                plt.ylabel("Feature space for the 2nd feature")
                plt.suptitle(clfs[i].__class__.__name__,
                             fontsize=14, fontweight='bold')
                clfs[i].fit(X,y)
                x1,x2=X[:,0].min()-1,X[:,0].max()+1
                y1,y2=X[:,1].min()-1,X[:,1].max()+1
                xx,xy=np.meshgrid(np.arange(x1,x2,0.1),np.arange(y1,y2,0.1))
                z=clfs[i].predict(np.c_[xx.ravel(),xy.ravel()])
                z=z.reshape(xx.shape)
                plt.contourf(xx, xy, z, cmap=cmap, alpha=al)
                plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap)
                plt.show()
    
    def plot_learning_curves_reg(model,X,y):
        X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2)
        train_errors , val_errors =[],[]
        for m in range(1,len(X_train)):
            model.fit(X_train[:m],y_train[:m])
            y_train_predict = model.predict(X_train[:m])
            y_val_predict = model.predict(X_val)
            train_errors.append(mean_squared_error(y_train_predict,y_train[:m]))
            val_errors.append(mean_squared_error(y_val_predict,y_val))
        plt.plot(np.sqrt(train_errors),"r-+",linewidth=2,label="train")
        plt.plot(np.sqrt(val_errors),"b-",linewidth=3,label="val")
        plt.legend(['Training error','validation error'])
                
    t0 ,t1 =5,50 #learning schedule hyperparameter
    def learning_schedule(t):
            return t0/(t+t1)
        
        
    def viz_sgd(X,y,n_epochs=10,m=1):
        X_b = np.c_[np.ones(X.shape),X]
        #Stochastic batch gradient descent
        n_epochs= n_epochs
        
        m=m
        theta = np.random.randn(2,1)  #random initialisation
        plt.scatter(X,y,color="blue")
        print('coefficient   Intercept')
        for epoch in range(n_epochs):
            for i in range(m):
                random_index=np.random.randint(m)
                xi =X_b[random_index:random_index+1]
                yi= y[random_index:random_index+1]
                gradients = 2*xi.T.dot(xi.dot(theta)-yi)
                eta=learning_schedule(epoch*m+i)
                theta=theta-eta*gradients
                y_plot= theta[0]+theta[1]*X
                plt.plot(X,y_plot,color="red")
                print(f"{theta[0]} - {theta[1]}")
                plt.xlim(-4,4)
        plt.show()
    
    
    def viz_reg(model,X,y,figure_size=(12,8),marker='*'):
        if type(model)!=list:
            model.fit(X,y)
            plt.figure(figsize=figure_size)
            plt.title("The visualization of the regression data.")
            plt.xlabel("Feature space for the 1st feature")
            plt.ylabel("Feature space for the 2nd feature")
            plt.suptitle(str(model).split('(')[0],
                    fontsize=14, fontweight='bold')
            x1,x2=X.min()-1,X.max()+1
            xx=np.array(np.arange(x1,x2,0.1))
            xx=xx.reshape((-1,1))
            xx.shape
            z=model.predict(xx)
            plt.scatter(X,y,marker=marker)
            plt.plot(xx,z,color='red')    
            plt.show()
        else:
            for mod in model:
                mod.fit(X,y)
                plt.figure(figsize=figure_size)
                plt.title("The visualization of the regression data.")
                plt.xlabel("Feature space for the 1st feature")
                plt.ylabel("Feature space for the 2nd feature")
                plt.suptitle(str(mod).split('(')[0],
                    fontsize=14, fontweight='bold')
                x1,x2=X.min()-1,X.max()+1
                xx=np.array(np.arange(x1,x2,0.1))
                xx=xx.reshape((-1,1))
                xx.shape
                z=mod.predict(xx)
                plt.scatter(X,y,marker=marker)
                plt.plot(xx,z,color='red')    
                plt.show()
    
    
    def viz_cluster(model,X,figure_size=(12,8),help_clusters=False,lower_range=2,upper_range=8):
        if help_clusters==True:
            range_n_clusters = np.arange(lower_range,upper_range,1)
            for n_clusters in range_n_clusters:
                # Create a subplot with 1 row and 2 columns
                fig, (ax1, ax2) = plt.subplots(1, 2)
                fig.set_size_inches(18, 7)
                
                # The 1st subplot is the silhouette plot
                # The silhouette coefficient can range from -1, 1 but in this example all
                # lie within [-0.1, 1]
                ax1.set_xlim([-0.1, 1])
                # The (n_clusters+1)*10 is for inserting blank space between silhouette
                # plots of individual clusters, to demarcate them clearly.
                ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
                
                # Initialize the clusterer with n_clusters value and a random generator
                # seed of 10 for reproducibility.
                clusterer = model.set_params(n_clusters=n_clusters)
                cluster_labels = clusterer.fit_predict(X)
                
                # The silhouette_score gives the average value for all the samples.
                # This gives a perspective into the density and separation of the formed
                # clusters
                silhouette_avg = silhouette_score(X, cluster_labels)
                print("For n_clusters =", n_clusters,
                      "The average silhouette_score is :", silhouette_avg)
                
                # Compute the silhouette scores for each sample
                sample_silhouette_values = silhouette_samples(X, cluster_labels)
                
                y_lower = 10
                for i in range(n_clusters):
                    # Aggregate the silhouette scores for samples belonging to
                    # cluster i, and sort them
                    ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]
                    
                    ith_cluster_silhouette_values.sort()
                    
                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i
                    
                    color = cm.nipy_spectral(float(i) / n_clusters)
                    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                      0, ith_cluster_silhouette_values,
                                      facecolor=color, edgecolor=color, alpha=0.7)
                    
                    # Label the silhouette plots with their cluster numbers at the middle
                    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                    
                    # Compute the new y_lower for next plot
                    y_lower = y_upper + 10  # 10 for the 0 samples
                    
                    ax1.set_title("The silhouette plot for the various clusters.")
                    ax1.set_xlabel("The silhouette coefficient values")
                    ax1.set_ylabel("Cluster label")
                    
                    # The vertical line for average silhouette score of all the values
                    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
                    
                    ax1.set_yticks([])  # Clear the yaxis labels / ticks
                    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
                    
                    # 2nd Plot showing the actual clusters formed
                    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
                    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                                c=colors, edgecolor='k')
                    
                    # Labeling the clusters
                    #centers = clusterer.cluster_centers_
                    # Draw white circles at cluster centers
                    #ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    #           c="white", alpha=1, s=200, edgecolor='k')
                    
                    #for i, c in enumerate(centers):
                     #   ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                     #              s=50, edgecolor='k')
                        
                    ax2.set_title("The visualization of the clustered data.")
                    ax2.set_xlabel("Feature space for the 1st feature")
                    ax2.set_ylabel("Feature space for the 2nd feature")
                    
                    plt.suptitle(("Silhouette analysis clustering on sample data "
                                  "with n_clusters = %d" % n_clusters),
                    fontsize=14, fontweight='bold')
                        
                    plt.show()
        else:
            y=model.fit_predict(X)
            if X.shape[1]!=2:
                raise TypeError('X should contain only 2 features')
            plt.figure(figsize=figure_size)
            plt.title("The visualization of the clustered data.")
            plt.xlabel("Feature space for the 1st feature")
            plt.ylabel("Feature space for the 2nd feature")
            plt.suptitle(model.__class__.__name__,
                    fontsize=14, fontweight='bold')
            z=model.fit_predict(X)
            plt.scatter(X[:, 0], X[:, 1], c=z, cmap='jet')
            plt.show()

class repo:
    def report(df):
        shape=f"Shape  {df.shape}"
        buf = io.StringIO()
        df.info(buf=buf)
        info=buf.getvalue()
        des=str(df.describe())
        type(des)
        values_cols=''    
        for col in df.columns:
            values_cols+=str(df[col].value_counts())
            values_cols+='\n'
        correlation_matrix=''
        correlation_matrix=str(df.corr())
        if len(correlation_matrix)==0:
            correlation_matrix='There are no numerical features in the dataset'
        pdf=FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=1)
        pdf.set_font("Arial",size=25,style='BIU')
        pdf.set_text_color(255,0,128)
        pdf.cell(200,10,txt="REPORT",border=2,ln=2,align='C')
        pdf.set_font("Arial",size=12)
        pdf.set_text_color(0,0,0)
        pdf.set_font(family='Times',style='BIU',size=15)
        pdf.multi_cell(0,10,txt=f'Shape')
        pdf.set_font(family='Arial',style='I',size=12)
        pdf.multi_cell(0,10,txt=f'{df.shape}',align='J')
        pdf.set_font(family='Times',style='BIU',size=15)
        pdf.multi_cell(0,10,txt=f'Names of the Columns')
        pdf.set_font(family='Arial',style='I',size=12)
        pdf.multi_cell(0,10,txt=f'{str(df.columns)}',align='J')
        pdf.set_font(family='Times',style='BIU',size=15)
        pdf.multi_cell(0,10,txt=f'Information')
        pdf.set_font(family='Arial',style='I',size=12)
        pdf.multi_cell(0,10,txt=f'{info}',align='J')
        pdf.set_font(family='Times',style='BIU',size=15)
        pdf.multi_cell(0,10,txt=f'Statistical Description')
        pdf.set_font(family='Arial',style='I',size=12)
        pdf.multi_cell(0,10,txt=f'{des}',align='J')
        pdf.set_font(family='Times',style='BIU',size=15)
        pdf.multi_cell(0,10,txt=f'Types of Instances')
        pdf.set_font(family='Arial',style='I',size=12)
        pdf.multi_cell(0,10,txt=f'{values_cols}',align='J')
        pdf.set_font(family='Times',style='BIU',size=15)
        pdf.multi_cell(0,10,txt=f'Correlation matrix')
        pdf.set_font(family='Arial',style='I',size=12)
        pdf.multi_cell(0,10,txt=f'{correlation_matrix}',align='J')
        # save the pdf with name .pdf 
        val=confirm(text='Do you want to save the pdf',title='Confirmation',buttons=['OK','Cancel'])
        if val=='OK':
            pdf_name=prompt(text='Give the name of the pdf file',title='NAME',default='Report')
            pdf.output(f"{pdf_name}.pdf")
            alert(text='Your pdf is stored in your current working directory', title='PDF saved!!!', button='OK')
        else:
            alert(text='Your pdf has not been saved', title='PDF not saved', button='OK')
    
