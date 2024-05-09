#!/usr/bin/env python
# coding: utf-8

# In[67]:
import os,csv,json
import struct
from numpy import *
from pathlib import Path
from Bio import SeqIO
import time
#datetime
from datetime import datetime
from tqdm import tqdm
import networkx as nx,numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plti
###########################################################################
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
###########################################################################
from sklearn.metrics import DetCurveDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
###########################################################################
from joblib import dump, load
from sklearn.metrics import precision_score,auc,precision_recall_curve,recall_score, confusion_matrix, classification_report,accuracy_score,f1_score,roc_auc_score,matthews_corrcoef, roc_curve






class kmer_featurization:

    def __init__(self, k):
        """
        seqs: a list of DNA sequences
        k: the "k" in k-mer
        """
        self.k = k
        self.letters = ['A', 'T', 'C', 'G']
        self.multiplyBy = 4 ** np.arange(k-1, -1, -1) # the multiplying number for each digit position in the k-number system
        self.n = 4**k # number of possible k-mers

    def obtain_kmer_feature_for_a_list_of_sequences(self, seqs, write_number_of_occurrences=False):
        """
        Given a list of m DNA sequences, return a 2-d array with shape (m, 4**k) for the 1-hot representation of the kmer features.
        Args:
          write_number_of_occurrences:
            a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.    
        """
        kmer_features = []
        for seq in seqs:
            this_kmer_feature = self.obtain_kmer_feature_for_one_sequence(seq.upper(), write_number_of_occurrences=write_number_of_occurrences)
            kmer_features.append(this_kmer_feature)

        kmer_features = np.array(kmer_features)

        return kmer_features

    def obtain_kmer_feature_for_one_sequence(self, seq, write_number_of_occurrences=False):
        """
        Given a DNA sequence, return the 1-hot representation of its kmer feature.
        Args:
          seq: a string, a DNA sequence write_number_of_occurrences:
            a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.
        """
        number_of_kmers = len(seq) - self.k + 1

        kmer_feature = np.zeros(self.n)

        for i in range(number_of_kmers):
            this_kmer = seq[i:(i+self.k)]
            this_numbering = self.kmer_numbering_for_one_kmer(this_kmer)
            kmer_feature[this_numbering] += 1

        if not write_number_of_occurrences:
            kmer_feature = kmer_feature / number_of_kmers

        return kmer_feature

    def kmer_numbering_for_one_kmer(self, kmer):
        """
        Given a k-mer, return its numbering (the 0-based position in 1-hot representation)
        """
        digits = []
        for letter in kmer:
            digits.append(self.letters.index(letter))

        digits = np.array(digits)

        numbering = (digits * self.multiplyBy).sum()

        return numbering


# In[18]:


def regenSeQ(L):
    s=L[0]
    for i,k in enumerate(L):
        if i>0:s+=k[-1]
    return s




# In[96]:



def loadData(obj,Root,fileN,dtp,dcount):
    ClassL=[];DataL=[]
    dataPath=f"{Root}/{dtp}/Dataset/{fileN}.tsv"
    with open(dataPath,'r') as f:
        cf=csv.reader(f, delimiter='\t')
        for i,r in enumerate(cf):
            if i>0:# and i<20:
                ClassL.append(int(r[1]))
                spar=r[0].split('[SEP]')
                A=spar[0].split();B=spar[1].split()
                seqA=regenSeQ(A);seqB=regenSeQ(B)
                A_fet=obj.obtain_kmer_feature_for_one_sequence(seqA, write_number_of_occurrences=False)
                B_fet=obj.obtain_kmer_feature_for_one_sequence(seqB, write_number_of_occurrences=False)
                DataL.append(np.concatenate((A_fet,B_fet)))
            if i==dcount:break
    D=np.array(DataL);Cl=np.array(ClassL)
    print(f'{fileN} :: Data dim: {D.shape},Pos:{sum(Cl)}, Neg:{Cl.shape[0]-sum(Cl)}, TotalC: {i}')
    return D,Cl
#print(len(ClassL))


# ### 1 :: DataLoad

# In[61]:


def loadDataSet(obj,Root,dtp,trC,tsC):
    '''
       Rnapol2 total : [train data: 3,00,000]  [test data: 15,000]
       CTCF total    : [train data: 5,00,000]  [test data: 20,000]
    '''
    #dtp='CTCF'#'RNAPOL2'#'CTCF'#  #'CTCF'
    fileTr='train';fileTs='dev';
    X_train, Y_train=loadData(obj,Root,fileTr,dtp,trC)
    X_test, Y_test=loadData(obj,Root,fileTs,dtp,tsC)
    # print('Data loaded\n')
    return X_train, Y_train,X_test, Y_test


# In[71]:


classifiers = {
  'SVM' : svm.SVC(C=2.0, kernel='rbf',gamma=0.0005, probability=True),
  'SVMi':OneVsRestClassifier(SVC(kernel='linear', probability=True), n_jobs=-1),
  'RF' : RandomForestClassifier(max_depth=5, n_estimators=250,min_samples_leaf =20),  # random_forest
  'KNN' : KNeighborsClassifier(n_neighbors=7),  # kneighbors
}
#OneVsRestClassifier(SVC(kernel='linear', probability=True), n_jobs=-1)
# 'SVMi':OneVsRestClassifier(SVC(kernel='linear', probability=True, class_weight='auto'), n_jobs=-1),
# In[77]:


def StatEvaluation(y_test,prTsd,probaScore):
    pre=precision_score(y_test, prTsd)
    rec=recall_score(y_test, prTsd)
    accu=accuracy_score(y_test, prTsd)
    f1=f1_score(y_test, prTsd)
    mcc=matthews_corrcoef(y_test,prTsd)
    aucS=roc_auc_score(y_test,probaScore[:, 1])
    fpr, tpr, thr = roc_curve(y_test,probaScore[:, 1])
    preRF, recRF, thrRF = precision_recall_curve(y_test, probaScore[:,1])
    auPRC= auc(recRF, preRF)
    PRC=[preRF, recRF]
    ARC=[fpr, tpr]
    return pre,rec,accu,f1,mcc,aucS,auPRC,PRC,ARC


# In[81]:

def GetMerge(M,N,ik):
    MD=tuple([M[i] for i in range(len(M)) if i!=ik])
    ND=tuple([N[i] for i in range(len(N)) if i!=ik])
    D=np.concatenate(MD,axis=0)
    L=np.concatenate(ND,axis=0)
    # print(D.shape,L.shape)
    return D,L

# In[82]:


def WriteRes(resDir,fnam,sl):
    f=open("{}/{}.csv".format(resDir,fnam),"w")
    f.write(sl);f.close()
def WritePred(resDir,fnam,CLA):
    #dumpD = json.dumps(CLA, cls=NumpyEncoder)
    dumpD = CLA.tolist()
    with open("{}/{}.json".format(resDir,fnam),"w") as f:
        json.dump(dumpD,f)


# In[98]:
# def RunMLy(kmr,Root,dtp,ml,TrnWholeD,DtX,DtY,n):

def RunMLfld(kmr,Root,dtp,MLname,TrnWholeD,DtX,DtY,TsD,TsL,n):
    CLSF=classifiers[MLname]
    sln="Classifier,Fold,AUC,auPRC,Accu,Pre,Recall,F1,MCC\n"
    rSL="Classifier,Fold,AUC,AUPRC,Accu,Pre,Recall,F1,MCC\n"
    hSL="Classifier,Fold,AUC,AUPRC,Accu,Pre,Recall,F1,MCC\n"
    modelDir=f"{Root}/{dtp}/KM-{kmr}/Model/{MLname}"
    resDir=f"{Root}/{dtp}/KM-{kmr}/Result/{MLname}"
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)
    if not os.path.exists(resDir):
        os.makedirs(resDir)
    rPR_ARC={};hPR_ARC={}
    for ind in tqdm(range(n)):
        # print(f' Fold-{ind} learning')
        test_datas=DtX[ind]; test_labels=DtY[ind]
        train_datas, train_labels=GetMerge(DtX,DtY,ind)
        CLSF.fit(train_datas,train_labels)
        dump(CLSF,f"{modelDir}/{MLname}_F{ind}.joblib")
        print(f"{dtp} Fold-{ind} model dumped")

        '''Prediction on Test data :Fold-wise'''
        prdLabel_tr=CLSF.predict(test_datas)
        prob_tr = CLSF.predict_proba(test_datas)
        Rpre,Rrec,Raccu,Rf1,Rmcc,RaucS,RauPRC,rPRC,rARC=StatEvaluation(test_labels,prdLabel_tr,prob_tr)
        rPR_ARC[ind]=[rPRC,rARC]
        rSL+='{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(MLname,ind,RaucS,RauPRC,Raccu,Rpre,Rrec,Rf1,Rmcc)
        WritePred(resDir,'TR_{}_F{}_prCL'.format(MLname,ind),prdLabel_tr)
        WritePred(resDir,'TR_{}_F{}_ACL'.format(MLname,ind),test_labels)
        print(f"{dtp} Fold-{ind} Test result dumped")


        '''Prediction on holdOut data :Fold-wise''' 
        prdLabel_ts=CLSF.predict(TsD)
        prob_ts = CLSF.predict_proba(TsD)
        Hpre,Hrec,Haccu,Hf1,Hmcc,HaucS,HauPRC,hPRC,hARC=StatEvaluation(TsL,prdLabel_ts,prob_ts)
        hPR_ARC[ind]=[hPRC,hARC]
        hSL+='{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(MLname,ind,HaucS,HauPRC,Haccu,Hpre,Hrec,Hf1,Hmcc)
        WritePred(resDir,'HO_{}_F{}_prCL'.format(MLname,ind),prdLabel_ts)
        print(f"{dtp} Fold-{ind} Hold-out result dumped")
        WriteRes(resDir,'TR_{}'.format(MLname),rSL)
        WriteRes(resDir,'HO_{}'.format(MLname),hSL)
    print('**********************************************************')
    print(f'Prediction with :: {dtp} Fold-wise Test data [Kmer:{kmr}]:: {MLname}::\n{rSL}')
    print('----------------------------------------------------')
    print(f'Prediction with :: {dtp} Fold-wise HoldOut data [Kmer:{kmr}]:: {MLname}::\n{hSL}')
    print('**********************************************************')
def RunMLwl(kmr,Root,dtp,MLname,TrnWholeD,TrD,TrL,TsD,TsL,n):
    CLSF=classifiers[MLname]
    sln="Classifier,Fold,AUC,auPRC,Accu,Pre,Recall,F1,MCC\n"
    rSL="Classifier,Fold,AUC,AUPRC,Accu,Pre,Recall,F1,MCC\n"
    hSL="Classifier,Fold,AUC,AUPRC,Accu,Pre,Recall,F1,MCC\n"
    modelDir=f"{Root}/{dtp}/KM-{kmr}/Model/{MLname}"
    resDir=f"{Root}/{dtp}/KM-{kmr}/Result/{MLname}"
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)
    if not os.path.exists(resDir):
        os.makedirs(resDir)
    #if TrnWholeD:
        # print(f"Learning on Whole data")
    CLSF.fit(TrD,TrL)
    prTsd=CLSF.predict(TsD)
    probaScore=CLSF.predict_proba(TsD)
    pre,rec,accu,f1,mcc,aucS,auPRC,PRC,ARC=StatEvaluation(TsL,prTsd,probaScore)
    # sl="WholeData:: Accuracy={:.3f},AUC={:.3f},auPRC={:.3f},Precision={:.3f},Recall={:.3f},F1={:.3f},MCC={:.3f}\n".        format(accu,aucS,auPRC,pre,rec,f1,mcc)
    hSL+='{},WholeD,{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(MLname,aucS,auPRC,accu,pre,rec,f1,mcc)
    WritePred(resDir,f'HO_{MLname}_WholeD_prCL',prTsd)
    WriteRes(resDir,f'HO_{MLname}_WholeD',hSL)
    print(f'Prediction with :: {dtp} [Wholedata] on HoldOut:: {MLname}:: \n {hSL}')
    
    WriteRes(resDir,'TR_{}'.format(MLname),rSL)
    WriteRes(resDir,'HO_{}'.format(MLname),hSL)
    print('**********************************************************')
    print(f'Prediction with :: {dtp} Fold-wise Test data [Kmer:{kmr}]:: {MLname}::\n{rSL}')
    print('----------------------------------------------------')
    print(f'Prediction with :: {dtp} Fold-wise HoldOut data [Kmer:{kmr}]:: {MLname}::\n{hSL}')
    print('**********************************************************')

'''
        Rnapol2 total : [train data: 3,00,000]  [test data: 15,000]
        CTCF total    : [train data: 5,00,000]  [test data: 20,000]
'''

def main():
    
    Root = Path(os.getcwd()).parent
    # print(Root)
    kmr = 3 
    TrnWholeD=True
    obj = kmer_featurization(kmr)

    # dtp='RNAPOL2'#'CTCF'#'CTCF'#  #'CTCF'
    
    dtpL=['RNAPOL2']#,'CTCF']#'CTCF'#  #'CTCF'
    dln={'RNAPOL2':{'tr':150000,'ts':15000},'CTCF':{'tr':250000,'ts':20000}}       ##  Train with 50%  data 
    #dln={'RNAPOL2':{'tr':300000,'ts':15000},'CTCF':{'tr':500000,'ts':20000}}      ##  Train with 100% data
    for dtp in dtpL:
        trC=dln[dtp]['tr'];tsC=dln[dtp]['ts']
        TrD,TrL,TsD,TsL=loadDataSet(obj,Root,dtp,trC,tsC)
        # print(TrD.shape,TrL.shape)

        MLAlg=['RF','KNN']#'SVMi']#,'RF','KNN']#,'SVM']
        begin_time=datetime.now()
        TRnDtype=[True,False]
        for TrnWholeD in TRnDtype:
            if TrnWholeD==False:
                n=5
                DtX=np.split(TrD,n,axis=0)
                DtY=np.split(TrL,n,axis=0)
                for ml in tqdm(MLAlg):
                    b_time=datetime.now()
                    print(f'{ml} processing at:{b_time}')
                    print('--------------------------------')
                    RunMLfld(kmr,Root,dtp,ml,TrnWholeD,DtX,DtY,TsD,TsL,n)

            else:
                n=0
                for ml in tqdm(MLAlg):
                    RunMLwl(kmr,Root,dtp,ml,TrnWholeD,TrD,TrL,TsD,TsL,n)

        end_time = datetime.now()
        train_time = (end_time-begin_time).total_seconds()
        print(f'{ml} training time :{train_time}')
        # break
        


if __name__ == '__main__':
    main()

'''
**********************************************************█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [06:00<00:00, 72.45s/it]
Prediction with :: RNAPOL2 Fold-wise Test data [Kmer:3]:: SVMi::
Classifier,Fold,AUC,AUPRC,Accu,Pre,Recall,F1,MCC
SVMi,0,0.7724,0.7864,0.6980,0.7453,0.5916,0.6597,0.4031
SVMi,1,0.7591,0.7814,0.6777,0.7312,0.5838,0.6493,0.3657
SVMi,2,0.7798,0.7998,0.6977,0.7476,0.6059,0.6693,0.4039
SVMi,3,0.7734,0.7882,0.6917,0.7485,0.5883,0.6588,0.3942
SVMi,4,0.7716,0.7808,0.6967,0.7345,0.5966,0.6584,0.3977

----------------------------------------------------
Prediction with :: RNAPOL2 Fold-wise HoldOut data [Kmer:3]:: SVMi::
Classifier,Fold,AUC,AUPRC,Accu,Pre,Recall,F1,MCC
SVMi,0,0.7476,0.7770,0.6770,0.7203,0.6035,0.6567,0.3612
SVMi,1,0.7473,0.7764,0.6760,0.7249,0.5918,0.6516,0.3609
SVMi,2,0.7479,0.7776,0.6790,0.7247,0.6016,0.6574,0.3658
SVMi,3,0.7483,0.7773,0.6800,0.7275,0.5996,0.6574,0.3684
SVMi,4,0.7478,0.7771,0.6830,0.7181,0.6270,0.6694,0.3707
**********************************************************
SVMi training time :360.341771

*********************************************************█████████████████████████████████████████████████████▌                          | 4/5 [08:03<01:59, 119.41s/it]
Prediction with :: RNAPOL2 Fold-wise Test data [Kmer:4]:: SVMi::
Classifier,Fold,AUC,AUPRC,Accu,Pre,Recall,F1,MCC
SVMi,0,0.7568,0.7764,0.6855,0.7493,0.5678,0.6460,0.3843
SVMi,1,0.7335,0.7487,0.6740,0.7052,0.5854,0.6398,0.3517
SVMi,2,0.7555,0.7774,0.6810,0.7319,0.5880,0.6521,0.3714
SVMi,3,0.7525,0.7755,0.6705,0.7162,0.5894,0.6466,0.3491
SVMi,4,0.7700,0.7951,0.6960,0.7536,0.6070,0.6724,0.4030

----------------------------------------------------
Prediction with :: RNAPOL2 Fold-wise HoldOut data [Kmer:4]:: SVMi::
Classifier,Fold,AUC,AUPRC,Accu,Pre,Recall,F1,MCC
SVMi,0,0.7297,0.7630,0.6670,0.7067,0.5977,0.6476,0.3404
SVMi,1,0.7301,0.7628,0.6660,0.7005,0.6074,0.6506,0.3369
SVMi,2,0.7301,0.7631,0.6670,0.7086,0.5938,0.6461,0.3410
SVMi,3,0.7299,0.7634,0.6700,0.7106,0.5996,0.6504,0.3466
SVMi,4,0.7305,0.7633,0.6690,0.7100,0.5977,0.6490,0.3447

**********************************************************
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [10:03<00:00, 120.64s/it]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [10:03<00:00, 603.19s/it]
SVMi training time :603.191529


'''
