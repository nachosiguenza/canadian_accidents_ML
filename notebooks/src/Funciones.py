#!/usr/bin/env python
# coding: utf-8

# In[3]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss


# In[ ]:


def plot_feature_importance(importance,names,model_type):
    
   #Create arrays from feature importance and feature names
   feature_importance = np.array(importance)
   feature_names = np.array(names)

   #Create a DataFrame using a Dictionary
   data={'feature_names':feature_names,'feature_importance':feature_importance}
   fi_df = pd.DataFrame(data)

   #Sort the DataFrame in order decreasing feature importance
   fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

   #Define size of bar plot
   plt.figure(figsize=(20,25))
   #Plot Searborn bar chart
   sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
   #Add chart labels
   plt.title(model_type + 'FEATURE IMPORTANCE')
   plt.xlabel('FEATURE IMPORTANCE')
   plt.ylabel('FEATURE NAMES')


# In[4]:


def cramers_v(z, t, df=None):
    x=df[z]
    y=df[t]
    confusion_matrix = pd.crosstab(x,y)
    confusion_matrix=confusion_matrix.values
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    c=np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
    print('La correlación de' ,z, 'y' ,t, 'es' ,c)


# In[5]:


def true_false(var,df=None):
    cols=['Col','Var','True', 'False']
    TF= pd.DataFrame(columns = cols)
    for x in df[var].unique():
        TF=TF.append({'Col':var,'Var':x, 'True':100*(df[var][df[var]==x][df.C_SEV==True].count())/df[var][df[var]==x].count(), 'False':100*(df[var][df[var]==x][df.C_SEV==False].count())/df[var][df[var]==x].count()}, ignore_index=True)
    TF.set_index('Col', inplace=True)
    TF=TF.sort_values(by=["True", "False"], ascending=False)
    return TF


# In[6]:


def count_tipo(var,df=None,grafico=False):
    cols=['Col','Var','count','perc']
    CT= pd.DataFrame(columns = cols)
    for x in df[var].unique():
        CT=CT.append({'Col':var,'Var':x, 'count':df[var][df[var]==x].count(), 'perc':df[var][df[var]==x].count()*100/df[var].count()}, ignore_index=True)
    CT.set_index('Col', inplace=True)
    CT=CT.sort_values(by=["count"], ascending=False)
    if grafico==True:
        print(CT)
        CT.head(10).plot.bar()
    else:
        return(CT)


# In[7]:


def porc_TF_mascomun(var,df=None,n=5):
    T=true_false(var,df)
    Z=count_tipo(var,df)
    T=T.merge(Z,on='Var').set_axis(T.index)
    T=T.sort_values(by=["count"], ascending=False)
    p=Z.head(n)
    return(T[T['Var'].isin(p.Var)])


# In[8]:


def na_var(var,df):
    z=df[var].isna().sum()
    print('\nvariable:' ,var, 'Número de na:',z)


# In[9]:


def dame_variables_categoricas2(dataset=None):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función dame_variables_categoricas:
    ----------------------------------------------------------------------------------------------------------
        -Descripción: Función que recibe un dataset y devuelve una lista con los nombres de las 
        variables categóricas
        -Inputs: 
            -- dataset: Pandas dataframe que contiene los datos
        -Return:
            -- lista_variables_categoricas: lista con los nombres de las variables categóricas del
            dataset de entrada
    '''
    if dataset is None:
        print(u'\nFaltan argumentos por pasar a la función')
        return 1
    lista_variables_categoricas = []
    other = []
    for i in dataset.columns:
        if (dataset[i].dtype!='float64') & (dataset[i].dtype!='int64') & (dataset[i].dtype!='bool'):
                lista_variables_categoricas.append(i)
        else:
                other.append(i)

    return lista_variables_categoricas, other


# In[10]:


def plot_feature(df,col_name, isContinuous):
    """
    Visualize a variable with and without faceting on the loan status.
    - col_name is the variable name in the dataframe
    - full_name is the full variable name
    - continuous is True if the variable is continuous, False otherwise
    """
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,3), dpi=90)
    

    if isContinuous:
        sns.histplot(df.loc[df[col_name].notnull(), col_name], kde=False, ax=ax1)
    else:
        sns.countplot(df[col_name], color='#5975A4', saturation=1, ax=ax1) # order=sorted(df[col_name].unique()),
    ax1.set_xlabel(col_name)
    ax1.set_ylabel('Count')
    ax1.set_title(col_name)
    plt.xticks(rotation = 90)

    if isContinuous:
        sns.boxplot(y=col_name, x='C_SEV', data=df, ax=ax2)
        ax2.set_ylabel(col_name)
        ax2.set_xlabel('')
        ax2.set_title(col_name + ' by C_SEV')
    else:
        data = df.groupby(col_name)['C_SEV'].value_counts(normalize=True).to_frame('proportion').reset_index()
        data.columns = [col_name, 'C_SEV', 'proportion']
        sns.barplot(x = col_name, y = 'proportion', hue= "C_SEV", data = data, saturation=1, ax=ax2)
        ax2.set_ylabel('C_SEV fraction')
        ax2.set_title('C_SEV')
        plt.xticks(rotation = 90)
        ax2.set_xlabel(col_name)
    
    plt.tight_layout()


# In[11]:


def get_corr_matrix(dataset = None, metodo='pearson', size_figure=[10,8]):
    # Para obtener la correlación de Spearman, sólo cambiar el metodo por 'spearman'

    if dataset is None:
        print(u'\nHace falta pasar argumentos a la función')
        return 1
    sns.set(style="white")
    # Compute the correlation matrix
    corr = dataset.corr(method=metodo) 
    # Set self-correlation to zero to avoid distraction
    for i in range(corr.shape[0]):
        corr.iloc[i, i] = 0
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=size_figure)
    # Draw the heatmap with the mask and correct aspect ratio
    #Generamos una máscara para evitar tener un espejo en la parte superior de la gráfica y simplificar la observación
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    
    sns.heatmap(corr, center=0,
                square=True, linewidths=.5, annot=True, cmap ='viridis', mask=mask) #cbar_kws={"shrink": .5}
    plt.show()
    
    return 0

