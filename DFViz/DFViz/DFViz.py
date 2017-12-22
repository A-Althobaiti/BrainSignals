import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.offline import plot, plot_mpl
from math import ceil
import igraph as ig
from plotly.offline import plot
from plotly.graph_objs import *
from .network_layout import network_layout

class Edge(object):
	def __init__(self, source, destination, weight):
		self.source = source
		self.destination = destination
		self.weight = weight

	def __str__(self):
		return '{}--({})-->{}'.format(self.source, self.weight, self.destination)

	def __le__(self, other):
		if isinstance(other, Edge):
			return self.weight <= other.weight
		return self.weight <= other

	def __lt__(self, other):
		if isinstance(other, Edge):
			return self.weight < other.weight
		return self.weight < other

	def __eq__(self, other):
		if isinstance(other, Edge):
			return self.weight == other.weight
		return self.weight == other

	def __ge__(self, other):
		if isinstance(other, Edge):
			return self.weight >= other.weight
		return self.weight >= other

	def __gt__(self, other):
		if isinstance(other, Edge):
			return self.weight > other.weight
		return self.weight > other

	def __repr__(self):
		return self.__str__()

class Vertex(object):
	def __init__(self, name):
		self.vertex = name
		self.group = None
		self.edges = []
		self.visited = False
		self.time_forward = -1
		self.time_backward = -1

	def add_edge(self, destination, weight):
		self.edges.append(Edge(self, destination, weight))

	def __str__(self):
		return self.vertex

	def __repr__(self):
		return self.__str__()

class Graph(object):
	def __init__(self):
		self.vertices = []
		self.edges = []
		self.mst = []

	def create_graph(self, matrix):
		assert(matrix.shape[0] == matrix.shape[1])
		distance_matrix = matrix.max() - matrix
		vertices = [Vertex('{}'.format(ch)) for ch in range(1, matrix.shape[0] + 1)]

		for r in range(matrix.shape[0]):
			for c in range(matrix.shape[0]):
				if c != r:
					vertices[r].add_edge(vertices[c], distance_matrix[r,c])

		return vertices

	def plot_ly(self):
		p = go.Scatter(
				x=[1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8],
				y=[1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4],
				mode='markers')
		
		trace = [p]
		for e in self.edges:
			x1 = int(int(e.source.vertex)%(len(self.vertices)/4))
			if x1 == 0: x1 = 8
			y1 = ceil(int(e.source.vertex)/(len(self.vertices)/4))
			x2 = int(int(e.destination.vertex)%(len(self.vertices)/4))
			if x2 == 0: x2 = 8
			y2 = ceil(int(e.destination.vertex)/(len(self.vertices)/4))
			trace.append(go.Scatter(x=[x1,x2],y=[y1,y2],mode='lines'))

		fig = go.Figure(data=trace)

		plot(fig, image='png', image_filename='image_file_name', output_type='file')

	@classmethod
	def from_matrix(cls, matrix):
		g = cls()
		g.vertices = g.create_graph(matrix)
		g.edges = [e for v in g.vertices for e in v.edges]
		return g

	@classmethod
	def from_edges(cls, edges):
		g = cls()

		names = []
		for e in edges:
			if e.source.vertex not in names: names.append(e.source.vertex)
			if e.destination.vertex not in names: names.append(e.destination.vertex)

		vertices = [Vertex(name) for name in names]

		for e in edges:
			for v in vertices:
				if e.source.vertex == v.vertex:
					for u in vertices:
						if e.destination.vertex == u.vertex:
							v.add_edge(u, e.weight)
							break
					break
		
		g.vertices = vertices
		g.edges = [e for v in g.vertices for e in v.edges]

		return g

	def prim(self):
		mst_vertices = [self.vertices[0]]
		mst_edges = []

		edge_queue = [e for e in mst_vertices[0].edges]
		edge_queue.sort(reverse=True)

		while len(mst_vertices) < len(self.vertices):
			edge = edge_queue.pop()
			if edge.destination not in mst_vertices:
				mst_edges.append(edge)
				mst_vertices.append(edge.destination)
				for e in edge.destination.edges:
					edge_queue.append(e)
				edge_queue.sort(reverse=True)

		self.mst = Graph.from_edges(mst_edges)

	def print(self):
		for v in self.vertices:
			print('{}'.format(v))
			for e in v.edges:
				print('{}'.format(e))
			print('-------------')

	def print_mst(self):
		for e in self.mst:
			print(e)

def reverse(g):
	return Graph.from_edges([Edge(e.destination, e.source, e.weight) for e in g.edges])

def SCC(g, epsilon=0):
	count = 0
	def DFS_visit(v):
		nonlocal count
		v.visited = True
		v.group = count
		for e in v.edges:
			if e.destination.visited == False:
				DFS_visit(e.destination)

	dfs_g = DFS(g,epsilon)
	rev_g = reverse(dfs_g)

	for v in dfs_g.vertices:
		for u in rev_g.vertices:
			if v.vertex == u.vertex:
				u.time_backward = v.time_backward

	rev_g.vertices.sort(key=lambda x: x.time_backward, reverse=True)
	
	for v in rev_g.vertices:
		if v.visited == False:
			DFS_visit(v)
			count = count + 1
	count = count + len(g.vertices) - len(rev_g.vertices)
	if count == 0:
		return len(g.vertices)
	return count

def DFS(g, epsilon=0):
	time = 1
	def DFS_visit(v):
		nonlocal time
		v.visited = True
		v.time_forward = time
		time = time + 1
		for e in v.edges:
			if e.destination.visited == False:
				DFS_visit(e.destination)
		v.time_backward = time
		time = time + 1

	g = Graph.from_edges([e for e in g.edges if e <= epsilon])
		
	for v in g.vertices:
		v.visited = False
	for u in g.vertices:
		if u.visited == False:
			DFS_visit(u)
	return g
	
def plot_SCC(g, plot=False):
	g.prim()
	thresholds = np.linspace(0, min(max(g.mst.edges).weight, 1), 1000)
	barcodes = [[] for _ in range(len(g.vertices))]
	for threshold in thresholds:
		barcodes[SCC(g, threshold) - 1].append(threshold)

	for i in range(len(g.vertices)-1, -1, -1):
		if len(barcodes[i]) >= 1:
			for j in range(i-1, -1, -1):
				if len(barcodes[j]) > 0:
					barcodes[i].append(barcodes[j][0])
					break
	barcodes[0].append(barcodes[0][-1] + barcodes[0][-1] * 0.05)

	if plot:
		for i in range(len(barcodes)):
			xs = barcodes[i]
			ys = [i+1] * len(xs)
			plt.plot(xs, ys, linewidth=2)

		plt.axis([0,barcodes[0][-1],0,len(g.vertices)])
		plt.show()

	return barcodes

def some_func_1(g, barcodes):
	threshold = max(barcodes, key=len)[0]
	edges = [e for e in g.edges if e < threshold]
	g_1 = Graph.from_edges(edges)
	v_original = [v.vertex for v in g.vertices]
	v_new = [v.vertex for v in g_1.vertices]
	v_missing = [v for v in v_original if v not in v_new]
	for v in v_missing:
		g_1.vertices.append(Vertex(v))
	return g_1

def print_cliques(g):
	count = 0
	while True:
		vs = [v for v in g.vertices if v.group == count]
		if len(vs) == 0:
			break
		print('Clique {}:\n{}'.format(count, vs))
		count +=1

def some_func(g, title, path, auto_open=True):
	N = len(g.vertices)
	L = len(g.edges)

	color = 'rgb(55,126,184)' if title == 'delta' else \
			'rgb(228,26,28)'  if title == 'theta' else \
			'rgb(77,175,74)'  if title == 'alpha' else \
			'rgb(255,127,0)'  if title == 'beta' else \
			'rgb(247,129,191)'  if title == 'gamma' else \
			'rgb(125,125,125)'

	Edges=[(int(e.source.vertex)-1,int(e.destination.vertex)-1) for e in g.edges]

	G=ig.Graph(Edges, directed=True)

	labels=['Ch ' + v.vertex for v in g.vertices]
	group=[0 for v in g.vertices]

	layt=G.layout('kk', dim=2)

	Xn=[layt[k][0] for k in range(N)]# x-coordinates of nodes
	Yn=[layt[k][1] for k in range(N)]# y-coordinates
	Zn=[1 for k in range(N)]# z-coordinates
	Xe=[]
	Ye=[]
	Ze=[]
	for e in Edges:
		Xe+=[layt[e[0]][0],layt[e[1]][0], None]# x-coordinates of edge ends
		Ye+=[layt[e[0]][1],layt[e[1]][1], None]
		Ze+=[1,1, None]

	trace1=Scatter3d(x=Xe,
				   y=Ye,
				   z=Ze,
				   mode='lines',
				   line=Line(color='rgb(125,125,125)', width=1),
				   hoverinfo='none'
				   )
	trace2=Scatter3d(x=Xn,
				   y=Yn,
				   z=Zn,
				   mode='markers',
				   name='actors',
				   marker=Marker(symbol='dot',
								 size=6,
								 color=color,
								 line=Line(color='rgb(50,50,50)', width=0.5)
								 ),
				   text=labels,
				   hoverinfo='text'
				   )
	axis=dict(showbackground=False,
			  showline=False,
			  zeroline=False,
			  showgrid=False,
			  showticklabels=False,
			  title=''
			  )
	layout = Layout(
			title=title,
			width=600,
			height=600,
			showlegend=False,
			scene=Scene(
			xaxis=XAxis(axis),
			yaxis=YAxis(axis),
			zaxis=ZAxis(axis),
			),
		margin=Margin(
			t=100
		),
		hovermode='closest',
		annotations=Annotations([
				Annotation(
				showarrow=False,
				text="Data source: Frostig Lab",
				xref='paper',
				yref='paper',
				x=0,
				y=0.1,
				xanchor='left',
				yanchor='bottom',
				font=Font(
				size=14
				)
				)
			]),	)
	dataframe=Data([trace1, trace2])
	fig=Figure(data=dataframe, layout=layout)
	plot(fig, filename='./plots/{}/{}.html'.format(path, title), auto_open=auto_open)

def plot_bands(bands, title):
	def generate_graph(g, z):
		N = len(g.vertices)
		L = len(g.edges)

		Edges=[(int(e.source.vertex)-1,int(e.destination.vertex)-1) for e in g.edges]

		G=ig.Graph(Edges, directed=True)

		labels=['Ch ' + v.vertex for v in g.vertices]
		group=[z for v in g.vertices]

		layt=G.layout('kk', dim=2)

		Xn=[layt[k][0] for k in range(N)]# x-coordinates of nodes
		Yn=[layt[k][1] for k in range(N)]# y-coordinates
		Zn=[z for k in range(N)]# z-coordinates
		Xe=[]
		Ye=[]
		Ze=[]
		for e in Edges:
			Xe+=[layt[e[0]][0],layt[e[1]][0], None]# x-coordinates of edge ends
			Ye+=[layt[e[0]][1],layt[e[1]][1], None]
			Ze+=[z,z, None]

		return Xn, Yn, Zn, Xe, Ye, Ze, group, labels

	Xn, Yn, Zn, Xe, Ye, Ze, group, labels = [],[],[],[],[],[],[],[]

	for band in bands:

		bandX, bandY, bandZ, bandXE, bandYE, bandZE, g, l = generate_graph(band[0], band[1])

		Xn += bandX
		Yn += bandY
		Zn += bandZ
		Xe += bandXE
		Ye += bandYE
		Ze += bandZE
		group += g
		labels += l

	trace1=Scatter3d(x=Xe,
				   y=Ye,
				   z=Ze,
				   mode='lines',
				   line=Line(color='rgb(125,125,125)', width=1),
				   hoverinfo='none'
				   )
	trace2=Scatter3d(x=Xn,
				   y=Yn,
				   z=Zn,
				   mode='markers',
				   name='actors',
				   marker=Marker(symbol='dot',
								 size=6,
								 color=group,
								 colorscale='Picnic',
								 line=Line(color='rgb(50,50,50)', width=0.5)
								 ),
				   text=labels,
				   hoverinfo='text'
				   )
	axis=dict(showbackground=False,
			  showline=False,
			  zeroline=False,
			  showgrid=False,
			  showticklabels=False,
			  title=''
			  )
	layout = Layout(
			title=title,
			width=800,
			height=700,
			showlegend=False,
			scene=Scene(
			xaxis=XAxis(axis),
			yaxis=YAxis(axis),
			zaxis=ZAxis(axis),
			),
		margin=Margin(
			t=100
		),
		hovermode='closest',
		annotations=Annotations([
				Annotation(
				showarrow=False,
				text="Data source: Frostig Lab",
				xref='paper',
				yref='paper',
				x=0,
				y=0.1,
				xanchor='left',
				yanchor='bottom',
				font=Font(
				size=14
				)
				)
			]),	)
	dataframe=Data([trace1, trace2])
	fig=Figure(data=dataframe, layout=layout)
	plot(fig)

def get_layout(g, z):
	N = len(g.vertices)
	L = len(g.edges)
	threshold = 'NaN'
	
	if len(g.edges) > 0:
		g.edges.sort(key=lambda x: x.weight, reverse=True)
		threshold = 1 - g.edges[0].weight

	Edges=[(int(e.source.vertex)-1,int(e.destination.vertex)-1) for e in g.edges]

	G=ig.Graph(Edges, directed=True)

	labels=['Ch ' + v.vertex for v in g.vertices]
	group=[z for v in g.vertices]

	layt=G.layout('kk', dim=2)

	Xn=[layt[k][0] for k in range(N)]# x-coordinates of nodes
	Yn=[layt[k][1] for k in range(N)]# y-coordinates
	Zn=[z for k in range(N)]# z-coordinates
	Xe=[]
	Ye=[]
	Ze=[]
	for e in Edges:
		Xe+=[layt[e[0]][0],layt[e[1]][0], None]# x-coordinates of edge ends
		Ye+=[layt[e[0]][1],layt[e[1]][1], None]
		Ze+=[z,z, None]

	return network_layout(Xn, Yn, Zn, Xe, Ye, Ze, group, labels, threshold)

def load_bands(path='./pre_dual_all/'):
	bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
	return {'{}_{}'.format(band_1, band_2):np.load('{}{}_{}.npy'.format(path, band_1, band_2)) for band_1 in bands for band_2 in bands}

def create_graphs(bands):
	keys = [k for k in bands.keys()]
	return {k:Graph.from_matrix(bands[k]) for k in keys}

def filter_graphs(graphs):
	keys = [k for k in graphs.keys()]
	return {k:some_func_1(graphs[k], plot_SCC(graphs[k])) for k in keys}

def create_classical_networks(filtered_graphs):
	layout = {}
	bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
	for level, band in enumerate(bands):
		layout[band + '_' + band] = get_layout(filtered_graphs[band + '_' + band], level+1)
	return layout

def create_dual_networks(filtered_graphs, classical_layout):
	layout = {}
	bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
	for band_1_level, band_1 in enumerate(bands):
		for band_2_level, band_2 in enumerate(bands):
			if band_1_level != band_2_level:
				g = filtered_graphs[band_1 + '_' + band_2]

				threshold = 'NaN'
				
				if len(g.edges) > 0:
					g.edges.sort(key=lambda x: x.weight, reverse=True)
					threshold = 1 - g.edges[0].weight

				edges = [(int(e.source.vertex)-1, int(e.destination.vertex)-1) for e in g.edges]
				Xe = []
				Ye = []
				Ze = []
				labels = []
				for s, d in edges:
					xs = classical_layout[band_1 + '_' + band_1].Xn[
						 classical_layout[band_1 + '_' + band_1].labels.index('Ch ' + str(s+1))]
					ys = classical_layout[band_1 + '_' + band_1].Yn[
						 classical_layout[band_1 + '_' + band_1].labels.index('Ch ' + str(s+1))]
					zs = band_1_level + 1
					
					xd = classical_layout[band_2 + '_' + band_2].Xn[
						 classical_layout[band_2 + '_' + band_2].labels.index('Ch ' + str(d+1))]
					yd = classical_layout[band_2 + '_' + band_2].Yn[
						 classical_layout[band_2 + '_' + band_2].labels.index('Ch ' + str(d+1))]
					zd = band_2_level + 1
					
					Xe+=[xs, xd, None]# x-coordinates of edge ends
					Ye+=[ys, yd, None]
					Ze+=[zs, zd, None]
					labels+=['Ch {} to Ch {}'.format(s+1, d+1)]
				layout[band_1 + '_' + band_2] = network_layout(None, None, None, Xe, Ye, Ze, None, labels, threshold)
	for band in bands:
		layout[band + '_' + band] = classical_layout[band + '_' + band]
	return layout

def plot_dual_networks(dual_networks, title):
	Xn, Yn, Zn, Xe, Ye, Ze, group, labels = [],[],[],[],[],[],[],[]	
	for net in dual_networks.keys():
		if dual_networks[net].Xn:
			Xn += dual_networks[net].Xn
		if dual_networks[net].Yn:
			Yn += dual_networks[net].Yn
		if dual_networks[net].Zn:
			Zn += dual_networks[net].Zn
		if dual_networks[net].group:
			group += dual_networks[net].group
		if dual_networks[net].labels:
			labels += dual_networks[net].labels

	nodes_traces = []
	bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
	for band in bands:
		net = band + '_' + band
		# plot the nodes first
		trace1=Scatter3d(x=dual_networks[net].Xn,
						 y=dual_networks[net].Yn,
						 z=dual_networks[net].Zn,
						 mode='markers',
						 name=band.capitalize(),
						 marker=Marker(symbol='dot',
									 size=6,
									 line=Line(color='rgb(50,50,50)', width=0.5)),
					   	text=dual_networks[net].labels,
					   	hoverinfo='text',
					   	legendgroup=band,
					   	showlegend=True)
		nodes_traces.append(trace1)
		

	edge_traces = []
	for net in dual_networks.keys():
		band_1, band_2 = net.split('_')
		if band_1 == band_2:
			name = band_1.capitalize()
		else:
			name = band_1.capitalize() + ' to ' + band_2.capitalize()
		trace2=Scatter3d(x=dual_networks[net].Xe,
					 y=dual_networks[net].Ye,
					 z=dual_networks[net].Ze,
				     mode='lines',
				     line=Line(color='rgb(125,125,125)', width=1),
				     hoverinfo='none',
				     name=name,
				     legendgroup=[band_1, band_2],
				     visible='legendonly')
		edge_traces.append(trace2)

	axis=dict(showbackground=False,showline=False,zeroline=False,showgrid=False,showticklabels=False,title='')
	layout = Layout(
			title=title,
			width=800,
			height=700,
			showlegend=True,
			scene=Scene(
			xaxis=XAxis(axis),
			yaxis=YAxis(axis),
			zaxis=ZAxis(axis),
			),
		margin=Margin(
			t=100
		),
		hovermode='closest',
		annotations=Annotations([
				Annotation(
				showarrow=False,
				text="Data source: Frostig Lab",
				xref='paper',
				yref='paper',
				x=0,
				y=0.1,
				xanchor='left',
				yanchor='bottom',
				font=Font(size=14)
				)
			]),	)
	dataframe=Data(nodes_traces + edge_traces)
	fig=Figure(data=dataframe, layout=layout)
	plot(fig)

def plot_network(g, epsilon):
	plt.subplot(212)
	plt.axis([0,9,0,5])

	for e in [e for e in g.edges if e < epsilon]:
		x1 = int( (int(e.source.vertex)-1) / 4) + 1
		y1 = 5 - (((int(e.source.vertex)-1) % 4) + 1)
		x2 = int( (int(e.destination.vertex)-1) / 4) + 1
		y2 = 5 - (((int(e.destination.vertex)-1) % 4) + 1)
		plt.plot([x1,x2],[y1,y2],'r',linewidth=0.7)

	xs=[1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8]
	ys=[1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4]
	plt.plot(xs,ys,'g^')

def fix_barcodes(barcodes):
	for i in range(len(barcodes)-1, -1, -1):
		if len(barcodes[i]) >= 1:
			for j in range(i-1, -1, -1):
				if len(barcodes[j]) > 0:
					barcodes[i].append(barcodes[j][0])
					break
	if(len(barcodes[0]) > 0):
		barcodes[0].append(barcodes[0][-1] + barcodes[0][-1] * 0.05)
	return barcodes

def plot_barcode(barcodes, xlim=1.05):
	plt.subplot(211)
	for i in range(len(barcodes)):
		xs = barcodes[i]
		ys = [i+1] * len(xs)
		plt.plot(xs, ys, linewidth=2)
	plt.axis([0,xlim,0,len(barcodes)+1])

def plot_evolution(g, destination, title=''):
	g.prim()
	thresholds = np.linspace(0, min(1,max(g.mst.edges).weight + max(g.mst.edges).weight * 0.05), 1000)
	barcodes = [[] for _ in range(len(g.vertices))]
	count = 1
	for threshold in thresholds:
		SCC_num = SCC(g, threshold) 
		barcodes[SCC_num - 1].append(threshold)
		barcodes = fix_barcodes(barcodes)
		plt.figure(1)
		plt.suptitle('{} - SCC: {} - threshold: {}'.format(title, SCC_num, threshold))
		plot_barcode(barcodes, min(1,max(g.mst.edges).weight + max(g.mst.edges).weight * 0.05))
		plot_network(g, threshold)
		plt.savefig(destination + '/' + str(count) + '.png')
		plt.close()
		print(count)
		count = count + 1

	return barcodes
