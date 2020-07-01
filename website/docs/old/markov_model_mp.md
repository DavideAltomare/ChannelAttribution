---
id: markov_model_mp
title: markov_model_mp
sidebar_label: markov_model_mp
---

### Description

Estimate a k-order Markov model from customer journey data. Differently from markov_model, this function iterates estimation until a desidered convergence is reached and enables multiprocessing.

### Usage

*markov_model_mp(Data, var_path, var_conv, var_value=None, var_null=None, order=1, nsim_start=100000.0, max_step=None, out_more=False, sep=>, ncore=1, nfold=10, seed=0, conv_par=0.05, rate_step_sim=1.5, verbose=True)*


### Parameters

**Data** : *DataFrame* : customer journeys.

**var_path**: *string* : column of Data containing paths.

**var_conv** : *string* : column of Data containing total conversions for each path.

**var_value** : *string, optional, default None* : column of Data containing revenue for each path.

**var_null** : *string* : column of Data containing total paths that do not lead to conversion.

**order** : *int, default 1* : Markov model order.

**nsim_start** : *int, default 1e5* : minimum number of simulations to be used in computation.

**max_step** : *int, default None* : maximum number of steps for a single simulated path. if NULL, it is the maximum number of steps found into Data.

**out_more** : *bool, default False* : if True, transition probabilities between channels and removal effects will be returned.

**sep** : *string, default ">"* : separator between the channels.

**ncore** : *int, default 1* : number of threads to be used in computation.

**nfold** : *int, default 10* : how many repetitions to be used to verify if convergence has been reached at each iteration.

**seed** : *int, default 0* : random seed. Giving this parameter the same value over different runs guarantees that results will not vary.

**conv_par** : *double, default 0.05* : convergence parameter for the algorithm. The estimation process ends when the percentage of variation of the results over different repetions is less than convergence parameter.

**rate_step_sim** : *double, default 0* : number of simulations used at each iteration is equal to the number of simulations used at previous iteration multiplied by rate_step_sim.

**verbose** : *bool, default True* : if True, additional information about process convergence will be shown.

### Returns

**list**

&nbsp;&nbsp;&nbsp;&nbsp;**result** : *Dataframe*

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**channel_name** : channel names

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**total_conversions** : conversions attributed to each channel

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**total_conversion_value** : revenues attributed to each channel

&nbsp;&nbsp;&nbsp;&nbsp;**transition_matrix** : *DataFrame*

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**channel_from**: channel from

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**channel_to** : channel to

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**transition_probability** : transition probability from channel_from to channel_to

&nbsp;&nbsp;&nbsp;&nbsp;**removal_effects** : *DataFrame*

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**channel_name** : channel names

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**removal_effects_conversion** : removal effects for each channel calculated using total conversions

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**removal_effects_conversion_value** : removal effects for each channel calculated using revenues

### Examples

Estimate a Makov model using total conversions

```python
markov_model_mp(Data, "path", "total_conversions")
```

Estimate a Makov model using total conversions and revenues

```python
markov_model_mp(Data, "path", "total_conversions", var_value="total_conversion_value")
```

Estimate a Makov model using total conversions, revenues and paths that do not lead to conversions

```python
markov_model_mp(Data, "path", "total_conversions", var_value="total_conversion_value", var_null="total_null")
```

Estimate a Makov model returning transition matrix and removal effects

```python
markov_model_mp(Data, "path", "total_conversions", var_value="total_conversion_value", var_null="total_null", out_more=TRUE)
```

Estimate a Markov model using 4 threads

```python
markov_model_mp(Data, "path", "total_conversions", var_value="total_conversion_value", ncore=4)
```