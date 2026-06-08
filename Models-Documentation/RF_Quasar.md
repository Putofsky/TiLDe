# Quasar Detection using Random forest
First we always work with light curves, so we need a way to extract information. Using statisticals tool we should be able to extract a lot of information from our data. The issue is that we can't know what are the statistical features that are relevant for our problem. So the idea was to use a feature extracter specialised for time Series data.
We are gonna use [[catch22]](#1) , using 22 features that were selected from a wide range of different statistical tools, used on a lot of differents problems. then they selected the best 22 features.

# The Data

We trained the RF two times to get 2 different models, one time we used true quasar and stars Data coming from gaia. The quasar object is the positive set, and the star is obviously the negative one.
We also used another set of data wich use synthetic quasars with true stars. The idea is that we might have issues with quasar detection if in our first training set is lacking quasar comportement diversity. We first used a [[DRW]](#2) model. But now we have better way if simulating Data wich is really close to true gaia data. The method is developped in the Data generating part.  
For our TrueData Dataset, we used 18.000 objects from gaia, 70% of them are Quasars and 30% of them are stars, we used 70% of the data as training sample, 15% as test sample and 15% as validation.


# Why a RF

We used [[catch22]](#1) to transform data into statisticall feature. The next step implied a CrossValidation 5 folds, to select what is the best model available to us. We tried different modelS; here are the results :

|  RF | XGB | XTree | KNN | Reg |
|:---:|:---:|:-----:|:---:|:---:|
|0,967|0,963| 0,963 |0.933|0.939|

The statistics obtened are in $\mathrm{BACC} = \frac{1}{2}\left(\frac{TP}{TP+FN} + \frac{TN}{TN+FP}\right)$ wich is a common tool used when the classes are unbalanced.
We can see that RF, XGB and XTree have really close results, and we choose RF because the CV gave us better results, but in practice this could not be the case, as some models can tend to learn better global caracteristics, so better generalisation, but as the results are given as a CV, we think that this kind of thinking is taking into account by the methology, and if a model is better than RF in practice, it's only due to luck.  


Why does this remark is important you may ask ? K-Fold CrossValisation is a method widely used in machine learning so the result should be good. But in fact K-Fold CV might not be the most accurate method for selecting a model, we don't understand well how the choice of subset impact it, and it's also impacted by the type of problem that we tacle [[3]](#3)...


# Training Phase

In training phase the model reach 97% BACC precision score. We also provide the following confusion matrix :

Matrice de confusion `(seuil = 0.73)`

| True label \ Predicted label | 0 | 1 |
|---|---:|---:|
| **0** | 🟦 **764** | 🟪 **10** |
| **1** | 🟪 **80** | 🟨 **1851** |

# An important analysis, feature analysis

To understand the results we  used feature permutation, this show relevancy of each features. 

# References

<a id="1">[1]</a> 
Carl H Lubba and .al, (2019). 
catch22: CAnonical Time-series CHaracteristics

<a id="2">[2]</a> 
Zu and .al, (2013).
IS QUASAR OPTICAL VARIABILITY A DAMPED RANDOM WALK?

<a id="3">[3]</a> 
Juan M Gorriz and .al, (2026)
Is K-fold cross validation the best model selection method for machine learning?
