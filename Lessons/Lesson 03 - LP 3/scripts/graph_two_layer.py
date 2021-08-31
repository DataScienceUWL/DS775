# function to show bipartite graph, not important to understand, loaded from graph_two_layer.py

import networkx as nx
import matplotlib.pyplot as plt

def graph_bipartite(left,right,edges={},labels={}):
    g = nx.Graph()
    g.add_nodes_from( left,bipartite=0)
    g.add_nodes_from(right,bipartite=1)

    if len(edges)>0:
        edges_new = edges
    else:
        edges_new = [(l,r) for l in left for r in right]
 
    for e in edges_new:
        g.add_edge(e[0],e[1])

    nleft = len(left)
    xw = .1
    eshift_left = .02
    delta_y = 1/(nleft+1)
    pos_left = { left[i]:(xw, 1-delta_y * (i+1)) for i in range(nleft) }
    pos_left_edge = { left[i]:(xw+eshift_left, 1-delta_y * (i+1)) for i in range(nleft) }
    xw = .1

    nright = len(right)
    xs = .9
    eshift_right = .015
    delta_y = 1/(nright+1)
    pos_right = { right[i]:(xs, 1-delta_y * (i+1)) for i in range(nright) }
    pos_right_edge = { right[i]:(xs-eshift_right, 1-delta_y * (i+1)) for i in range(nright) }

    plt.rcParams['figure.figsize'] = [9,9]
    plt.axis('off')

    nx.draw_networkx_nodes( left, pos_left,nodelist= left,node_color='r',node_size=0,alpha=0.3)
    nx.draw_networkx_nodes(right,pos_right,nodelist=right,node_color='b',node_size=0,alpha=0.3)

    pos = {}
    pos.update(pos_left)
    pos.update(pos_right)

    nx.draw_networkx_labels(g,pos,font_size=10,font_family='sans-serif')

    pos_e = {}
    pos_e.update(pos_left_edge)
    pos_e.update(pos_right_edge)
    
    nx.draw_networkx_edges(g,pos_e,edgelist=edges_new,width=1,alpha=0.4,edge_color='b')
    
    if len(labels)>0:        
        label_dict = dict(zip(edges_new,labels))
        nx.draw_networkx_edge_labels(g,pos_e,edge_labels=labels,font_color='red',font_size = 9, label_pos=.85)