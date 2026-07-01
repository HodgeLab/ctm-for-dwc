# Ramp Flow Estimation
A fundamental issue in traffic flow modeling is the existence of missing ramp flows, which are critical boundary conditions for freeway traffic simulation.
For example, in the Cell Transmission Model (CTM), on- and off-ramp flows are used to determine the on-ramp demand and off-ramp split ratio at each cell.
The existence of missing ramp flow data may result in a transportation network that is not fully observable, which is a severe challenge to overcome when modeling traffic flow in the network.

The estimation of missing ramp flow data is commonly categorized into one of two types:

1. Historical data is 100% lacking for the ramp of interest
2. Some historical data exists

For type 2, it may be possible to exploit temporal dependencies in the historical data to fill in, or impute, the missing values. This is commonly achieved through either a statistical method (e.g., ARIMA) or a machine learning-based method (e.g., recurrent neural network).
Type 1 is a more challenging problem, which has been addressed in [Muralidharan and Horowitz 2009](https://doi.org/10.3141/2099-07), [Kan et al. 2021](https://doi.org/10.1109/TITS.2020.2989365), and [Zhang et al. 2024](https://doi.org/10.1109/TITS.2023.3315693).

Furthermore, Type 1 may be further classified on the basis of ramp configuration.
Consider the three options presented in Figure 1 from Zhang et al. 2024.

![ramp_configs](ramp_configurations.png)

When we say we are concerned with estimating missing ramp flows, we are interested in determining the quantities $r$ and $s$, and we assume that the quantities $q^{up}$ and $q^{down}$ are known.
For stretches of type (a) or (b), we use conservation of flow to determine the missing ramp flow, setting $s$ or $r$ to zero depending on the type.

$$ q^{up} + r = q^{down} + s $$

For stretches of type (c), it is impossible to uniquely determine the pair of missing ramp flows from mainline flow alone.
This is the problem that Kan et al. 2021 and Zhang et al. 2024 address.

## Implementation Plan
We require a standalone ramp flow estimation module which is capable of generating CSV files for CTM cell demand and split ratios that are compatible with the CTM simulation engine.
Based on the classifications of missing ramp flow estimation previously discussed, the module is implemented as follows:

```python
if n_historical_points > threshold:
    temporal_fill()
elif (cell_type == TYPE_A) or (cell_type == TYPE_B):
    conservation_fill()
else:
    pair_fill()
```

For each vehicle detector station (VDS) associated with a ramp, we will determine whether the number of historical points in the available timeseries for the VDS(`n_historical_points`) exceeds some user-defined `threshold`.
If so, we will employ a fill strategy which exploits the temporal dependency of the historical data.
If not, we will evaluate the cell with which the VDS is associated and determine the configuration type as previously defined.
If the configuration type permits missing ramp flow estimation by conservation of flow, then that method will be used.
Finally, if the ramp flow data cannot be described by the prior methods, we will implement a fill strategy consistent with the approaches described by Kan et al. and Zhang et al.

Note that we make the problem specific to missing VDS estimation here. This is because a given cell may have multiple VDSs assigned to its on-ramp and/or off-ramp, and it is the *total* demand from these VDSs with which we are concerned, per the CTM Module's [Step 5](ctm_module.md#step-5-manual-validation-of-cell-mappings).