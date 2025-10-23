# Component Descriptions

This document describes the structure and attributes of the components of the CTM.

The traffic network in the CTM is represented as a directed graph using the [DiGraph](https://networkx.org/documentation/stable/reference/classes/digraph.html) class in the NetworkX library for Python. Links and nodes are represented using the [General Modeling Network Specification (GMNS)](https://github.com/zephyr-data-specs/GMNS), which helps to define directionality and connectivity in the network. 
The Python library [`osm2gmns`](https://osm2gmns.readthedocs.io/en/stable/) is essential for 
defining the GMNS representation such that it aligns with the real-world transportation network. 

# Links

Links represent road segments in the network. The network should be constructed such that links are
homogenous in lane count, grade, and geometrical features. Furthermore, each link should be
associated with at least one vehicle detector station (VDS) with macroscopic traffic data. These 
requirements enable each link to be represented by a fundamental diagram which relates observed 
traffic densities to observed flows for a particular point on the road. The fundamental diagrams
are calibrated to extract the free-flow speed, congestion wave speed, capacity, and jam density. 

Three kinds of links are allowed:
1. Mainline
2. Source
3. Sink

Mainline links connect two nodes, and are used to represent sections of the roadway that do not 
contain any on- or off-ramps. Source links represent on-ramps, and are used to introduce new traffic to the network. Conversely, Sink links represent off-ramps, and are used to accept traffic moving out of the network. Free flow traffic conditions are assumed to prevail at all boundaries of the freeway. 

Links hold specific attributes, which are categorized as either `static`, meaning the attribute is 
fixed for the duration of the simulation, or `dynamic`, meaning the attribute is updated at some
cadence throughout the simulation.

## Static Attributes

Static attributes are constant throughout the simulation. Attributes in this category include:

* Station ID
* Name
* Type
* Sensor Type
* Fwy
* Direction
* District
* County
* City
* HOV
* MS ID
* IRM
* CA PM
* Abs PM
* Length
* geometry
* capacity
* free_flow_speed
* congestion_wave_speed
* jam_density

## Dynamic Attributes

Dynamic attributes are allowed to change throughout the simulation. Attributes in this category
include:

* Status
* samples
* pct_observed
* total_flow_[veh/5-min]
* avg_occupancy_[%]
* avg_speed_[mph]
* good_samples_lane_i
* flow_lane_i
* avg_occupancy_lane_i
* avg_speed_lane_i
* observed_lane_i


