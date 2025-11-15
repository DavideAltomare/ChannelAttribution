---
id: channelattribution-functions
title: ChannelAttribution — Python API Reference
sidebar_label: Python API
description: API reference for the ChannelAttribution functions exposed from the Cython module.
---

This page documents the Python-callable functions defined in the Cython module.

> Source file: `ChannelAttribution.pyx`

## `__heuristic_models_1(vector[string] vy, vector[unsigned long int] vc, vector[double] vv, string sep)`
No documentation available.


---

## `__choose_order_1(vector[string] vy, vector[unsigned long int] vc, vector[unsigned long int] vn, unsigned long int max_order, string sep, unsigned long int ncore, unsigned long int roc_npt)`
No documentation available.


---

## `__markov_model_1(vector[string] vy, vector[unsigned long int] vc, vector[double] vv, vector[unsigned long int] vn, unsigned long int order, unsigned long int nsim_start,  unsigned long int max_step, unsigned long int out_more, string sep, unsigned long int ncore, unsigned long int nfold, unsigned long int seed, double conv_par, double rate_step_sim, int verbose)`
No documentation available.


---

## `__transition_matrix_1(vector[string] vy, vector[unsigned long int] vc, vector[unsigned long int] vn, unsigned long int order, string sep, int flg_equal)`
No documentation available.


---

## `heuristic_models(Data,var_path,var_conv,var_value=None, sep=">", flg_pro=True)`
No documentation available.


---

## `choose_order(Data,var_path,var_conv,var_null,max_order=10,sep=">",ncore=1,roc_npt=100,plot=True, flg_pro=True)`
No documentation available.


---

## `markov_model(Data,var_path,var_conv,var_value=None,var_null=None,order=1,nsim_start=1e5,max_step=None,out_more=False,sep=">",ncore=1, nfold=10, seed=0, conv_par=0.05,rate_step_sim=1.5,verbose=True, flg_pro=True)`
No documentation available.


---

## `transition_matrix(Data,var_path,var_conv,var_null,order=1,sep=">",flg_equal=True, flg_pro=True)`
No documentation available.


---

## `auto_markov_model(Data, var_path, var_conv, var_null, var_value=None, max_order=10, roc_npt=100, plot=False, nsim_start=1e5, max_step=None, out_more=False, sep=">", ncore=1, nfold=10, seed=0, conv_par=0.05, rate_step_sim=1.5, verbose=True, flg_pro=True)`
No documentation available.


---

## `install_pro()`
No documentation available.


---

## `ensure_package(import_name: str, pip_name: str | None = None, version: str | None = None)`
No documentation available.


---

## `http_post_form(url: str, form: dict, timeout: int = 300)`
No documentation available.


---

## `__init__(self)`
No documentation available.


---

## `handle_starttag(self, tag, attrs)`
No documentation available.


---

## `list_dir_files(dir_url)`
No documentation available.


---

## `resolve_pkg_url(pkg_value)`
No documentation available.


---

## `pip_install(url, extra_args=None)`
No documentation available.


---

## `get_system_info_dict()`
No documentation available.


---

## `_which_compiler()`
No documentation available.


---

## `_compiler_version(exe)`
No documentation available.


---

## `get_system_info()`
No documentation available.

