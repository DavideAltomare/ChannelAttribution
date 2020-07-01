---
id: heuristic_models
title: heuristic_models
sidebar_label: heuristic_models
---

### Description

Estimate three heuristic models (first-touch, last-touch and linear) from customer journey data.

### Usage

*heuristic_models(Data, var_path, var_conv, var_value=None, sep=>)*


### Parameters

**Data** : *DataFrame* : customer journeys.

**var_path**: *string* : column of Data containing paths.

**var_conv** : *string* : column of Data containing total conversions for each path.

**var_value** : *string, optional, default None*: column of Data containing revenue for each path.

**sep** : *string, default ">"*: separator between the channels.

### Returns

**DataFrame**

&nbsp;&nbsp;&nbsp;&nbsp;**channel_name** : channel names

&nbsp;&nbsp;&nbsp;&nbsp;**first_touch_conversions** : conversions attributed to each channel using first touch attribution.

&nbsp;&nbsp;&nbsp;&nbsp;**first_touch_value** : revenues attributed to each channel using first touch attribution.

&nbsp;&nbsp;&nbsp;&nbsp;**last_touch_conversions** : conversions attributed to each channel using last touch attribution.

&nbsp;&nbsp;&nbsp;&nbsp;**last_touch_value** : revenues attributed to each channel using last touch attribution.

&nbsp;&nbsp;&nbsp;&nbsp;**linear_touch_conversions** : conversions attributed to each channel using linear attribution.

&nbsp;&nbsp;&nbsp;&nbsp;**linear_touch_value** : revenues attributed to each channel using linear attribution.

### Examples

Estimate heuristic models on total conversions

```python
heuristic_models(Data,"path","total_conversions")
```

Estimate heuristic models on total conversions and total revenues

```python
heuristic_models(Data,"path","total_conversions",var_value="total_conversion_value")
```