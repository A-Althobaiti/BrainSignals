from DFViz import DFViz

data = DFViz.load_bands(path='./tests/post/')
print('Loading tests complete')
graphs = DFViz.create_graphs(data)
print('Creating graphs completed')
filtered = DFViz.filter_graphs(graphs)
print('Filtering graphs completed')
classical = DFViz.create_classical_networks(filtered)
print('Classical coherence completed')
dual_net = DFViz.create_dual_networks(filtered, classical)
print('Dual coherence completed')
DFViz.plot_dual_networks(dual_net, 'Plot Test')
print('Plotting completed')
