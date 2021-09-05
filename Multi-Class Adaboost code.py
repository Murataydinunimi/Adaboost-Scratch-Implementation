#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


def sign(y):
    x=[]
    for i in y:
        if i>0:
            x.append(1)
        elif i<0:
            x.append(-1)
        else:
            x.append(0)
    return(np.mat(x).T)


# In[124]:


def train_test_split(df,test_size,label_col,random_state=50):
    import random
    import pandas as pd
    
    random.seed(random_state)
    
    
    label_col = str(label_col)
    dat_len  = len(df)
    X= df.drop(columns=label_col)
    Y = df[label_col]
  
    
    
    if isinstance(test_size,float):
        test_size = round(test_size*dat_len)
        
    indices = list(data.index)
    test_indices = random.sample(population=indices, k=test_size)
    
    X_train = X.drop(test_indices)
    x_test  = X.loc[test_indices]
   
    Y_train= Y.drop(test_indices)
    y_test = Y.loc[test_indices]
   
    
    return X_train.sample(frac=1,random_state=random_state),x_test,Y_train.sample(frac=1,random_state=random_state),y_test


# In[4]:


def Decision_stump(data, label_set,P_i):
    data = np.mat(data)
    row_len,col_len = np.shape(data)
    label_set = np.mat(label_set).T
    predictions = np.ones((np.shape(data)[0],1))
    #search_size = search_size
    lowest_err = float('inf')
    decision_stump = {}
    for col in range(col_len):
        search_size= np.unique(data[:,col],axis=0)
        if len(search_size) > 100:
            search_size = round(np.sqrt(len(search_size)))
            #search_size= len(np.unique(data[:,col],axis=0)) 
            #search_size=round(search_size*0.6)
            col_min = data[:,col].min()
            col_max = data[:,col].max()
            stepSize = (col_max - col_min) / search_size
            for j in range(int(search_size)):
                    for theta_inq in ['lower', 'greater']:
                        theta = (col_min + float(j) * stepSize)
                        if theta_inq =='lower':   
                            predictions[data[:,col] <= theta ]= -1
                            est_mat = predictions.copy()
                            errors = np.mat(np.ones((row_len,1)))
                            errors[predictions == label_set] = 0
                            weighted_Error = P_i.T * errors
                            predictions = np.ones((np.shape(data)[0],1))

                            if weighted_Error < lowest_err:
                                lowest_err = weighted_Error
                                est_matrix = est_mat.copy()
                                decision_stump['dimen'] = col
                                #decision_stump['theta'] = round(theta,2)
                                decision_stump['theta'] = theta
                                decision_stump['theta_inq'] = theta_inq
                    else:

                        predictions[data[:,col]> theta ]= -1
                        est_mat = predictions.copy()
                        errors = np.mat(np.ones((row_len,1)))
                        errors[predictions == label_set] = 0 
                        weighted_Error = P_i.T * errors
                        predictions = np.ones((np.shape(data)[0],1))

                        if weighted_Error < lowest_err:
                            lowest_err = weighted_Error
                            est_matrix = est_mat.copy()
                            decision_stump['dimen'] = col
                            #decision_stump['theta'] = round(theta,2)
                            decision_stump['theta'] = theta
                            decision_stump['theta_inq'] = theta_inq

        else:
            for theta in search_size:
                for theta_inq in ['lower', 'greater']:
                    if theta_inq =='lower':   
                        predictions[data[:,col] <= theta ]= -1
                        est_mat = predictions.copy()
                        errors = np.mat(np.ones((row_len,1)))
                        errors[predictions == label_set] = 0
                        weighted_Error = P_i.T * errors
                        predictions = np.ones((np.shape(data)[0],1))

                        if weighted_Error < lowest_err:
                            lowest_err = weighted_Error
                            est_matrix = est_mat.copy()
                            decision_stump['dimen'] = col
                            #decision_stump['theta'] = round(theta,2)
                            decision_stump['theta'] = theta
                            decision_stump['theta_inq'] = theta_inq
                    else:

                        predictions[data[:,col]> theta ]= -1
                        est_mat = predictions.copy()
                        errors = np.mat(np.ones((row_len,1)))
                        errors[predictions == label_set] = 0 
                        weighted_Error = P_i.T * errors
                        predictions = np.ones((np.shape(data)[0],1))

                        if weighted_Error < lowest_err:
                            lowest_err = weighted_Error
                            est_matrix = est_mat.copy()
                            decision_stump['dimen'] = col
                            #decision_stump['theta'] = round(theta,2)
                            decision_stump['theta'] = theta
                            decision_stump['theta_inq'] = theta_inq

    return decision_stump,lowest_err,est_matrix


# In[5]:


def Adaboost(data,label_set,T_rounds):

    data = np.mat(data)
    models = []    
    row_len = data.shape[0]   
    P_i = np.mat(np.ones(shape=(row_len, 1)) / row_len) # define probability distribution
    preds = np.mat(np.zeros(shape=(row_len, 1))) 
    for i in range(T_rounds):
        decision_stump, epsilon, estimation = Decision_stump(data, label_set, P_i)  
        alpha = float(0.5 * np.log((1 - epsilon) / max(epsilon, 1e-16)))   
        decision_stump['alpha'] = alpha
        models.append(decision_stump)  
        P_i = np.multiply(P_i, np.exp(np.multiply(-1 * alpha * np.mat(label_set).T, estimation)))
        P_i =P_i / P_i.sum()   #normalize
        preds += alpha * estimation  #prediction
        totalerror = np.multiply(sign(preds) != np.mat(label_set).T, np.ones(shape=(row_len, 1)))  
        errorRate = totalerror.sum() / row_len 
        #print("total error: "+ "round" , str(i), errorRate,sep=' ')
        if errorRate == 0.0:
            break
    return models, preds


# In[6]:



def Adaboost_Predict(data,models):
    data = np.mat(data)
    row_len = data.shape[0]
    preds = np.mat(np.zeros(shape =(row_len, 1))) 
    for model in models:
        if model['theta_inq']=='lower':
            res = np.ones((row_len,1))
            res[data[:,model['dimen']] <= model['theta']] = -1
            preds += res*model['alpha']
        else:
            res = np.ones((row_len,1))
            res[data[:,model['dimen']]  > model['theta']] = -1
            preds += res*model['alpha']
            
    return(preds)


# In[7]:


def Adaboost_Multi_CV(data,k_folds,T_rounds,label_col,random_state):
    
    random.seed(random_state)
    
    X_train,x_test,Y_train,y_test = train_test_split(data,test_size=0.2,label_col=label_col,random_state=random_state)
    indices = np.array_split(list(X_train.index),k_folds)

    Cv_test_acc = np.mat(np.ones(shape=(len(T_rounds),k_folds)))
    Cv_train_acc = np.mat(np.ones(shape=(len(T_rounds),k_folds)))

    for T in T_rounds:
        for k in range(k_folds):
            train_preds= np.mat(np.ones(shape=(np.shape(X_train.drop(indices[k]))[0],len(np.unique(data[label_col])))))
            test_preds = np.mat(np.ones(shape=(np.shape(indices[k])[0],len(np.unique(data[label_col])))))
            for classes in range(len(np.unique(data[label_col]))):
                model,pred = Adaboost(X_train.drop(indices[k]),np.where(Y_train.drop(indices[k])==classes+1,1,-1),T)
                train_preds[:,classes]=np.multiply(train_preds[:,classes],pred)
                test_est = Adaboost_Predict(X_train.loc[indices[k]],model)
                test_preds[:,classes] = np.multiply(test_preds[:,classes],test_est)

            train_predictions = np.argmax(train_preds,axis=1)+1
            training_error =np.where(train_predictions!=np.mat(Y_train.drop(indices[k])).T,1,0).sum()
            Cv_train_acc[T_rounds.index(T),k]=1-(training_error/len(train_predictions))
            test_prediction = np.argmax(test_preds,axis=1)+1
            test_error = np.where(test_prediction!=np.mat(Y_train.loc[indices[k]]).T,1,0).sum()
            Cv_test_acc[T_rounds.index(T),k]=1-(test_error/len(test_prediction))
        print( 'Cv for',T,'rounds is completed')
    
    return(Cv_train_acc,Cv_test_acc)
        
     


# In[130]:


def learning_curve(data,k_folds,T,test_size,label_col,random_state):
    
    if isinstance(test_size,int):
                raise TypeError('Test_size should be a range of numbers')

    random.seed(random_state)
    
    Cv_test_acc = np.mat(np.ones(shape=(len(test_size),k_folds)))
    Cv_train_acc = np.mat(np.ones(shape=(len(test_size),k_folds)))
    testing_size =[]
    for value in test_size:
        X_train,x_test,Y_train,y_test = train_test_split(data,test_size=value,label_col=label_col,random_state=random_state)
        indices = np.array_split(list(X_train.index),k_folds)
        testing_size.append(value*len(data))
        for k in range(k_folds):
            train_preds= np.mat(np.ones(shape=(np.shape(X_train.drop(indices[k]))[0],len(np.unique(data[label_col])))))
            test_preds = np.mat(np.ones(shape=(np.shape(indices[k])[0],len(np.unique(data[label_col])))))
            for classes in range(len(np.unique(data[label_col]))):
                model,pred = Adaboost(X_train.drop(indices[k]),np.where(Y_train.drop(indices[k])==classes+1,1,-1),T)
                train_preds[:,classes]=np.multiply(train_preds[:,classes],pred)
                test_est = Adaboost_Predict(X_train.loc[indices[k]],model)
                test_preds[:,classes] = np.multiply(test_preds[:,classes],test_est)
            train_predictions = np.argmax(train_preds,axis=1)+1
            training_error =np.where(train_predictions!=np.mat(Y_train.drop(indices[k])).T,1,0).sum()
            Cv_train_acc[test_size.index(value),k]=1-(training_error/len(train_predictions))
            test_prediction = np.argmax(test_preds,axis=1)+1
            test_error = np.where(test_prediction!=np.mat(Y_train.loc[indices[k]]).T,1,0).sum()
            Cv_test_acc[test_size.index(value),k]=1-(test_error/len(test_prediction))
        print( 'Cv for',value,'Test size is completed')
    
 
    return(Cv_train_acc,Cv_test_acc,testing_size)
        
     
        
        
        


# In[9]:


from sklearn.datasets import make_classification
X,y = make_classification(n_samples=1000, n_features=3, n_informative=3,
                          n_redundant=0, n_repeated=0, n_classes=3,class_sep=1.1,
                          flip_y=0, random_state=50)

#manipulate the dataset so that it has the proper form for the usage of above functions.

data= np.column_stack((X,y))
data = pd.DataFrame.from_records(data)
data.columns = data.columns.astype(str)
data = data.rename(columns={'3':'decision'})
data['decision'] = data['decision']+1
data


# In[10]:


train,test = Adaboost_Multi_CV(data,k_folds=5,T_rounds=[10,20,30,40,50],
                               label_col='decision',random_state=50)


# In[11]:


np.mean(test,axis=1)


# In[12]:


np.mean(train,axis=1)


# In[13]:


train_acc,test_acc,test_size = learning_curve(data,k_folds=5,T=50,
                            test_size=[0.1,0.2,0.3,0.7,0.8],label_col='decision',random_state=50)


# In[14]:


np.mean(train_acc,axis=1)


# In[15]:


np.mean(test_acc,axis=1)


# In[16]:


test_size


# In[17]:


X_train,x_test,Y_train,y_test = train_test_split(data,test_size=0.2,label_col='decision',random_state=50)
train_preds= np.mat(np.ones(shape=(np.shape(X_train)[0],len(np.unique(data['decision'])))))
test_preds =  np.mat(np.ones(shape=(np.shape(x_test)[0],len(np.unique(data['decision'])))))
for classes in range(len(np.unique(data['decision']))):
    model,pred = Adaboost(X_train,np.mat(np.where(Y_train==int(classes)+1,1,-1)),50)
    train_preds[:,classes]=np.multiply(train_preds[:,classes],pred)
    test_est = Adaboost_Predict(x_test,model)
    test_preds[:,classes]=np.multiply(test_preds[:,classes],test_est)

train_predictions = np.argmax(train_preds,axis=1)+1 # because indexing starts from 0
training_error = np.where(train_predictions != np.mat(Y_train).T,1,0).sum()
test_prediction = np.argmax(test_preds,axis=1)+1
test_error = np.where(test_prediction != np.mat(y_test).T,1,0).sum()
print('test accuracy:',1-(test_error/len(x_test)))
print('training accuracy:',1-(training_error/len(X_train)))


# In[18]:


np.mean(test,axis=1)-np.std(test,axis=1)


# In[19]:


model


# In[22]:



data = pd.read_csv('forest-cover-type.csv')
data = data.drop(columns=['Id'])
forest = data.copy()


# In[23]:


forest.info()


# In[24]:


forest.isna().sum()


# In[25]:


figure(figsize=(12, 8), dpi=80)
sns.countplot(x="Cover_Type", data=forest)


# In[26]:


cv_train,cv_test = Adaboost_Multi_CV(forest,k_folds=5,T_rounds=[30,60,90,130,180,270,350,450],
                               label_col='Cover_Type',random_state=50)


# In[64]:


np.mean(cv_test,axis=1)


# In[33]:


np.mean(cv_train,axis=1)


# In[63]:


train_score_mean = np.mean(np.array(cv_train),axis=1)
train_score_std = np.std(np.array(cv_train),axis=1)
test_score_mean = np.mean(np.array(cv_test),axis=1)
test_score_std = np.std(np.array(cv_test),axis=1)
T_rounds = np.array([30,60,90,130,180,270,350,450])
plt.figure(figsize=(12, 8), dpi=80)
plt.title('Adaboost vs number of rounds')
plt.grid()
plt.fill_between(T_rounds, train_score_mean - train_score_std,
                 train_score_mean + train_score_std, alpha=0.1,
                 color="r")
plt.fill_between(T_rounds, test_score_mean - test_score_std,
                 test_score_mean + test_score_std, alpha=0.1, color="g")
plt.plot(T_rounds, train_score_mean, 'o-', color="r",
         label="Training score")
plt.plot(T_rounds, test_score_mean, 'o-', color="g",
         label="CV accuracy")
plt.legend()
plt.xlabel('Number of Rounds')
plt.ylabel('Accuracy')


# In[131]:


train_acc,test_acc,test_size = learning_curve(forest,k_folds=5,T=30,
                            test_size=[0.1,0.2,0.3,0.7,0.8,0.9],label_col='Cover_Type',random_state=50)


# In[132]:


np.mean(test_acc,axis=1)


# In[135]:


test_size=[0.1,0.2,0.3,0.7,0.8,0.9]
train_acc_mean = np.mean(np.array(train_acc),axis=1)
train_acc_std = np.std(np.array(train_acc),axis=1)
test_acc_mean = np.mean(np.array(test_acc),axis=1)
test_acc_std = np.std(np.array(test_acc),axis=1)
training_size = np.flip(np.array([int(i*len(forest)) for i in test_size]))
plt.figure(figsize=(12, 8), dpi=80)
plt.title('Training-Size vs 30-rounds Adaboost')
plt.grid()
plt.fill_between(training_size, train_acc_mean - train_acc_std,
                 train_acc_mean + train_acc_std, alpha=0.1,
                 color="r")
plt.fill_between(training_size, test_acc_mean - test_acc_std,
                 test_acc_mean + test_acc_std, alpha=0.1, color="g")
plt.plot(training_size, train_acc_mean, 'o-', color="r",
         label="Training score")
plt.plot(training_size, test_acc_mean, 'o-', color="g",
         label="CV accuracy")
plt.legend()
plt.xlabel('Training Size')
plt.ylabel('Accuracy')


# In[127]:


models=[]
X_train,x_test,Y_train,y_test = train_test_split(forest,test_size=0.1,label_col='Cover_Type',random_state=50)
train_preds= np.mat(np.ones(shape=(np.shape(X_train)[0],len(np.unique(forest['Cover_Type'])))))
test_preds =  np.mat(np.ones(shape=(np.shape(x_test)[0],len(np.unique(forest['Cover_Type'])))))
for classes in range(len(np.unique(forest['Cover_Type']))):
    model,pred = Adaboost(X_train,np.mat(np.where(Y_train==int(classes)+1,1,-1)),350)
    models.append(model)
    train_preds[:,classes]=np.multiply(train_preds[:,classes],pred)
    test_est = Adaboost_Predict(x_test,model)
    test_preds[:,classes]=np.multiply(test_preds[:,classes],test_est)

train_predictions = np.argmax(train_preds,axis=1)+1 # because indexing starts from 0
training_error = np.where(train_predictions != np.mat(Y_train).T,1,0).sum()
test_prediction = np.argmax(test_preds,axis=1)+1
test_error = np.where(test_prediction != np.mat(y_test).T,1,0).sum()
print('test accuracy:',1-(test_error/len(x_test)))
print('training accuracy:',1-(training_error/len(X_train)))


# In[ ]:


models=[]
X_train,x_test,Y_train,y_test = train_test_split(forest,test_size=0.1,label_col='Cover_Type',random_state=50)
train_preds= np.mat(np.ones(shape=(np.shape(X_train)[0],len(np.unique(forest['Cover_Type'])))))
test_preds =  np.mat(np.ones(shape=(np.shape(x_test)[0],len(np.unique(forest['Cover_Type'])))))
for classes in range(len(np.unique(forest['Cover_Type']))):
    model,pred = Adaboost(X_train,np.mat(np.where(Y_train==int(classes)+1,1,-1)),250)
    models.append(model)
    train_preds[:,classes]=np.multiply(train_preds[:,classes],pred)
    test_est = Adaboost_Predict(x_test,model)
    test_preds[:,classes]=np.multiply(test_preds[:,classes],test_est)

train_predictions = np.argmax(train_preds,axis=1)+1 # because indexing starts from 0
training_error = np.where(train_predictions != np.mat(Y_train).T,1,0).sum()
test_prediction = np.argmax(test_preds,axis=1)+1
test_error = np.where(test_prediction != np.mat(y_test).T,1,0).sum()
print('test accuracy:',1-(test_error/len(x_test)))
print('training accuracy:',1-(training_error/len(X_train)))


# In[97]:


features = np.unique([d['dimen'] for model in models for d in model],return_counts=True)[0]
counts = np.unique([d['dimen'] for model in models for d in model],return_counts=True)[1]
counts = pd.DataFrame(counts)
features =pd.DataFrame(features)
frames = [features,counts]
dat_to_plot=pd.concat(frames,axis=1)
dat_to_plot.columns = ['dim','count']
dat_to_plot=dat_to_plot.sort_values(by='count')
figure(figsize=(12, 8), dpi=80)
sns.barplot(x='dim',y='count',data=dat_to_plot, order=dat_to_plot['dim'],color='g')
plt.title("Most used features")
plt.ylabel('Counts')
plt.xlabel('Features')


# In[115]:


from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve, learning_curve,GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
data = pd.read_csv('forest-cover-type.csv')
X=data.drop(columns=['Cover_Type','Id'])
Y=data['Cover_Type']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=50)


T_rounds = [30,60,90,130,180,270,350,450]
k_folds=5
Cv_test_acc = np.mat(np.ones(shape=(len(T_rounds),k_folds)))

indices = np.array_split(list(X_train.index),k_folds)
for T in T_rounds:
    for k in  range(k_folds):
        ovo=OneVsRestClassifier(AdaBoostClassifier(n_estimators=T,algorithm='SAMME.R'))
        ovo.fit(X_train.drop(indices[k]),y_train.drop(indices[k]))
        y_pred = ovo.predict(X_train.loc[indices[k]])
        score = accuracy_score(y_train.loc[indices[k]],y_pred)
        Cv_test_acc[T_rounds.index(T),k]=score

np.mean(Cv_test_acc,axis=1)


# In[120]:


ovo=OneVsRestClassifier(AdaBoostClassifier(n_estimators=130,algorithm='SAMME.R'))
ovo.fit(X_train,y_train)
y_pred = ovo.predict(X_test)
score = accuracy_score(y_test,y_pred)
score


# In[121]:


T_rounds = [30,60,90,130,180,270,350,450]
k_folds=5
Cv_test_acc = np.mat(np.ones(shape=(len(T_rounds),k_folds)))

indices = np.array_split(list(X_train.index),k_folds)
for T in T_rounds:
    for k in  range(k_folds):
        ovo=OneVsRestClassifier(AdaBoostClassifier(n_estimators=T,algorithm='SAMME'))
        ovo.fit(X_train.drop(indices[k]),y_train.drop(indices[k]))
        y_pred = ovo.predict(X_train.loc[indices[k]])
        score = accuracy_score(y_train.loc[indices[k]],y_pred)
        Cv_test_acc[T_rounds.index(T),k]=score

np.mean(Cv_test_acc,axis=1)


# In[122]:


ovo=OneVsRestClassifier(AdaBoostClassifier(n_estimators=270,algorithm='SAMME'))
ovo.fit(X_train,y_train)
y_pred = ovo.predict(X_test)
score = accuracy_score(y_test,y_pred)
score


# In[ ]:




