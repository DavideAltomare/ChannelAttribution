

<center>
<img src="https://channelattribution.io/img/logo/2021_ChannelAttribution_AltHorizontal.svg" width="70%" height="auto" />
</center>
<br />
<br />


*What is multi-touch attribution?*
=================================

Multi-touch attribution is a method of marketing measurement that considers all touchpoints on the customer journey and allocates a certain amount of credit to each channel. This allows marketers to understand the value that each touchpoint contributes to driving a conversion. For instance, suppose a consumer is contemplating the purchase of a new smartphone. After conducting some research, they come across ads. Initially, they encounter a display ad, which they overlook. Subsequently, they notice a native ad on their Instagram feed, capturing their attention and redirecting them to a website. Finally, they receive a promotional offer via email, complete with a discount code that motivates them to make the purchase.

Each of these ads represents a touchpoint in the buyer's journey. With multi-touch attribution, marketers can examine the impact of the native ad and the email campaign, attributing the sale to these specific efforts. Meanwhile, they can recognize that the display ad was ineffective and make adjustments accordingly.

There are various multi-touch attribution models available to marketers, analyzing user-level data such as clicks and impressions to understand the influence of these events on the ultimate goal. These models vary in their evaluation of ad effectiveness, offering marketers diverse perspectives to shape their strategies.

*Multi-touch attribution models*
================================

Multi-touch attribution models aim to assign credit to different touchpoints in the customer journey. There are two families of attribution models: heuristic models and probabilistic models.

<h2>1. Multi-Touch Attribution Heuristic Models:</h2>

Heuristic models follow predefined rules to allocate credit to various touchpoints. These rules are often based on assumptions or industry best practices.
Simplicity: Heuristic models are generally simpler and easier to implement. They use straightforward rules to distribute credit, making them more accessible for marketers who may not have extensive statistical expertise.
Transparency: These models provide clear and understandable rules for assigning credit, offering transparency in the decision-making process. Examples of Heuristic Models are: 

- <b>First touch Attribution</b>: Attributes all credit to the first touchpoint in the customer journey, emphasizing the initial interaction's role in driving the conversion.

- <b>Last touch Attribution</b>: Attributes all credit to the last touchpoint in the customer journey, highlighting the final interaction as the most influential in the conversion process.

- <b>Linear Attribution</b>: Distributes credit equally to all touchpoints


<h2>2. Probabilistic Attribution Models:</h2>

Probabilistic models use statistical techniques and algorithms to analyze data and determine the probability of each touchpoint contributing to the conversion.
Complexity: These models are often more complex than heuristic models, as they involve analyzing large datasets and considering various factors to determine credit allocation.
Customization: Probabilistic models can be customized based on specific business requirements and nuances, allowing for a more tailored and accurate representation of the customer journey.
Examples of Probabilistic Models:

- <b>Markov Chain Models</b>: Analyzes the sequence of touchpoints to determine the transition probabilities between them.

- <b>Algorithmic Models</b>: Utilizes machine learning algorithms to predict the likelihood of each touchpoint influencing a conversion.


<i>Key Considerations:</i>

- <b>Flexibility</b>: Probabilistic models are often more flexible and adaptive to different business scenarios, while heuristic models may be less flexible but easier to implement.
- <b>Accuracy</b>: Probabilistic models are generally considered more accurate because they are data-driven and can adapt to changes in consumer behavior over time.
- <b>Resource Requirements</b>: Implementing and maintaining probabilistic models may require more resources, both in terms of data infrastructure and expertise, compared to heuristic models.

In summary, while heuristic models rely on predefined rules for credit allocation, probabilistic models leverage data-driven approaches and statistical techniques to provide a more nuanced and adaptable understanding of the customer journey. The choice between the two depends on factors such as the complexity of the business, the availability of data, and the desired level of accuracy and customization.


*ChannelAttribution*
====================

ChannelAttribution is a Python and R library that employs a k-order Markov representation to identify structural correlations in customer journey data. Additionally, the library incorporates three heuristic algorithms (first-touch, last-touch, and linear-touch approaches) to tackle the same problem. These algorithms are implemented in C++ and are well-suited for handling large-scale problems involving hundreds or thousands of channels.

<h2>Installation</h2>

<h3>Python</h3>


<b>NOTE</b>: Only Python3 is supported! Note! Only Python3 is supported! <b>Installation on Windows</b> requires [Microsoft Visual C++ 14.0 or greater](https://aka.ms/vs/stable/vs_BuildTools.exe).

<b>From PyPi</b>

```bash
pip install --upgrade setuptools
pip install Cython
pip install ChannelAttribution
```


<h2>Usage</h2>

In the following example we will show how Python library [ChannelAttribution](https://channelattribution.io) can be used to perform multi-touch attribution using heuristic models and Markov model.

First of all we need to load ChannelAttribution and data containing customer journeys:

```python

import numpy as np
import pandas as pd
from ChannelAttribution import *
import plotly.io as pio

Data = pd.read_csv("https://channelattribution.io/csv/Data.csv",sep=";")

```

Now we can performe attribution with heuristic models:

```python

H=heuristic_models(Data,"path","total_conversions",var_value="total_conversion_value")

```

We can also performe attribution with Markov model:

```python

M=markov_model(Data, "path", "total_conversions", var_value="total_conversion_value")

```

Now we can merge attributed conversion for each method:

```python

R=pd.merge(H,M,on="channel_name",how="inner")
R1=R[["channel_name","first_touch_conversions","last_touch_conversions",\
"linear_touch_conversions","total_conversions"]]
R1.columns=["channel_name","first_touch","last_touch","linear_touch","markov_model"]
R1=pd.melt(R1, id_vars="channel_name")

data = [dict(
  type = "histogram",
  histfunc="sum",
  x = R1.channel_name,
  y = R1.value,
  transforms = [dict(
    type = "groupby",
    groups = R1.variable,
  )],
)]

```

and plot them

```python

fig = dict({"data":data})
pio.show(fig,validate=False)

```

<img src="https://app.channelattribution.net/img/python_total_conversion.png" width="100%" height="auto" >
<br />

We can also get the same plot in terms of attributed conversion value:

```python

R2=R[["channel_name","first_touch_value","last_touch_value",\
"linear_touch_value","total_conversion_value"]]
R2.columns=["channel_name","first_touch","last_touch","linear_touch","markov_model"]

R2=pd.melt(R2, id_vars="channel_name")

data = [dict(
  type = "histogram",
  histfunc="sum",
  x = R2.channel_name,
  y = R2.value,
  transforms = [dict(
    type = "groupby",
    groups = R2.variable,
  )],
)]

```

<img src="https://app.channelattribution.net/img/python_total_value.png" width="100%" height="auto" >
<br />
<br />

Transaction level attribution problem
=====================================

Transaction level attribution is problematic using Markov models. Markov model can be considered a "global" approach because it performs attribution considering all the paths together. Instead heuristic models are "local" approaches because they perform attribution considering one path at the time and then global attribution for each channel is obtained through aggregation.  Using Markov model one can not go from global to local attribution in a unique way. Because Markov model returns the aggregate result and you can not go from this aggregation to single path attribution.

Why Markov model is a global approach? Markov model aggregates real paths to build a Markov graph, a graphical representation of a transition matrix. Markov graph is a mathematical representation of the dynamics between channels.  Markov model generates millions of random paths from Markov graph. Random paths are used to calculate importance weights (removal effects) for each channel. At the end of the process importance weights are normalized and multiplied by total conversions (the overall conversions observed for all the paths considered) to perform attribution for each channel. Thus Makov model is global because first it aggregates paths to build transition matrix and then it works with simulate paths “forgetting” each single real path.

For this reason, the choice of the right measure to perform transaction-level attribution is crucial and removal effects should not be used as shown in this  [article](https://medium.com/@davide.altomare/multi-touch-attribution-and-budget-allocation-ce04b492604d).


<h2>ChannelAttribution Pro</h2>

If you are interested in transaction level attribution and additional features related to attribution and budget allocation have a look to [ChannelAttribution Pro](https://channelattribution.io).

