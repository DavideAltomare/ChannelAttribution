---
id: choose_order
title: choose_order
sidebar_label: choose_order
---

### Description

Find the minimum Markov Model order that gives a good representation of customers’ behaviour for data considered. It requires paths that do not lead to conversion as input. Minimum order is found maximizing a penalized area under ROC curve.

### Usage

*choose_order(Data, var_path, var_conv, var_null, max_order=10, sep=>, ncore=1, roc_npt=100, plot=True)*


### Parameters

**Data** : *DataFrame* : customer journeys.

**var_path**: *string* : column of Data containing paths.

**var_conv** : *string* : column of Data containing total conversions for each path.

**var_null** : *string* : column of Data containing total paths that do not lead to conversion.

**max_order** : *int, default 10* : maximum Markov Model order to be considered.

**sep** : *string, default ">"* : separator between the channels.

**ncore** : *int, default 1* : number of threads to be used in computation.

**roc_npt**: *int, default 100* : number of points to be used for the approximation of roc curve.

**plot**: *bool, default True* : if TRUE, a plot with penalized auc with respect to order will be displayed.

### Returns

**list**

&nbsp;&nbsp;&nbsp;&nbsp;**roc** : *list of DataFrame* one for each order considered

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **tpr**: true positive rate

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **fpr**: false positive rate

&nbsp;&nbsp;&nbsp;&nbsp;**auc** : *DataFrame* with the following columns

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **order**: markov model order

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **auc**: area under the curve

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **pauc**: penalized auc

&nbsp;&nbsp;&nbsp;&nbsp;**suggested order** : *int* : estimated best order

### Examples

Estimate best makov model order for your data

```python
choose_order(Data, var_path="path", var_conv="total_conversions", var_null="total_null")
```