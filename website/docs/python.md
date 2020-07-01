
### def `transition_matrix(Data, var_path, var_conv, var_null=None, order=1, sep=>, flg_equal=True)`

Estimate a k-order transition matrix from customer journey data.

Parameters

Data : DataFrame
customer journeys.
var_path: string
column of Data containing paths.
var_conv : string
column of Data containing total conversions for each path.
var_null : string, default None
column of Data containing total paths that do not lead to conversion.
order : int, default 1
Markov model order.
sep : string, default ">"
separator between the channels.
flg_equal: bool, default True
if TRUE, transitions from a channel to itself will be considered.

Returns

list of DataFrames
channels: Dataframe
(column) id_channel : channel ids
(column) channel_name : channel names
transition_matrix : DataFrame
(column) channel_from: id channel from
(column) channel_to : id channel to
(column) transition_probability : transition probability from channel_from to channel_to

Examples

Estimate a third-order transition matrix using total conversions

>>> transition_matrix(Data, "path", "total_conversions", order=3)

Estimate a second-order transition matrix using total conversions and paths that do not lead to conversion

>>> transition_matrix(Data, "path", "total_conversions", var_null="total_null", order=2)


