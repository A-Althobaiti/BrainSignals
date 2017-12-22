# Brain Signals Dual-Frequency Visualization (DFViz)
This library has been developed to extend the visualization of the BrainSignal library. The BrainSignal library utilizes a novel approach of computing the dual-frequency cross-correlation. After cross-correlation is computed, this library can be utilized to visualize the interaction between populations of neurons on different frequencies.

## Prerequisites 

The following libraries are required for full functionality

```
Numpy
Plotly
Matplotlib
iGraph
```

## Installation

There is no automatic of installing the library yet, but it can be installed manually as explained below.

The following files:

```
DFViz.py
network_layout.py
```

has to be located in the same folder, and then you can start using it as explained below.

## Getting started

In order to visualize the dual-frequency interaction, the computed cross-correlation matrix has to be loaded through `load_bands()`. It must be a numpy an `n x n` array.
The `load_bands()` function takes one argument called path, where it specifies a folder containing Numpy matrices. An example of such a folder can be found under `./tests/post/` The `tester.py` file utilzes this folder as indicated below:

```
data = DFViz.load_bands(path='./tests/post/')
```

Given this matrix, a graph for each band can be constructed

```
graphs = DFViz.create_graphs(data)
```

To apply the (PH-Dowker) fliteration

```
filtered = DFViz.filter_graphs(graphs)
```

Now, the filtered graphs have been computed, we can get the layout
of the classical coherence as well as dual coherence. For the dual coherence
layout, the classical layout has to be computed first

```
classical = DFViz.create_classical_networks(filtered)
dual_net = DFViz.create_dual_networks(filtered, classical)
```

The dual_net variable from above is a dictionary with the names 
of all dual and classical coherences. Each value in the dictionary 
is an object of network_layout where Xn, Yn, and Zn attributes can be 
used in any 3d plot platform to visualize the classical coherence.
On the other hand, Xe, Ye, and Ze are attributes to draw edges for both
the classical and dual-frequency coherence.

Regardless of what visualization tool one might use, we are supporting
a PlotLy Network Visualization to visualze all the networks in one plot
```
DFViz.plot_dual_networks(dual_net, 'Plot Title')
```

## Testing
You may run `tester.py` as an example of how to run and use this toolbox.

## Deployment
A fully deployed version can be found [here](http://ec2-52-14-35-109.us-east-2.compute.amazonaws.com/bs/).

## Authors
- Abdulrahman Althobaiti

## License
This project is licensed under the MIT License
