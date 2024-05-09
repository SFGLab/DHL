#  DHL : Deep hybrid learning  for chromatin loop prediction

![image](https://github.com/SFGLab/DHL/assets/43639164/73bb2a7c-c9be-40f2-9807-a638f334f644)



This repository presents a deep hybrid learning (DHL) model that merges deep learning with traditional machine learning techniques to enhance the analysis of the human genome's spatial organization. Utilizing DNABERT, a deep learning network based on the BERT language model, alongside classical algorithms like support vector machines (SVMs), random forests (RFs), and K-nearest neighbors (KNN), this project aims to accurately predict the outcomes of spatial experiments such as Hi-C and ChIA-PET. The results demonstrate that DNABERT effectively predicts these experiments with high precision and that the DHL approach improves performance metrics on CTCF and RNAPII datasets. This innovative approach, while straightforward in concept, significantly enhances the understanding of genomic functions and disease development, making it a vital tool for researchers in the field of genomics.

Necessary script to train and predict the dataset are given in scripts directory.


          1)     DHL-train.py

          2)     DHL-predict.py
          
The pretrain model used in DHL is incorporated from the DNABERT (https://github.com/jerryji1993/DNABERT)
