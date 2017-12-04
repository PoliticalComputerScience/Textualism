import plotly
import plotly.plotly as py
from plotly.graph_objs import *
from neighborhood import *
from nltk.cluster.util import cosine_distance

#working example
"""
from plotly.graph_objs import Scatter, Layout

#when using jupyter notebooks, uncomment the following line and change function call below to plotly.offline.iplot
#plotly.offline.init_notebook_mode(connected=True)

plotly.offline.plot({
    "data": [Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1])],
    "layout": Layout(title="hello world")
})
"""


#coords should be a dictionary from a word to a 3d position
def generate_network(main_word, coords):

    #next step: generate edges.
    #   In general, we can modify or optionize this however we want to get whatever edge structures we find interesting. 
    #   --We might, for example, create more Neighborhoods (one for each "similar" word) to see if we get any connections within the 1-level-away words.
    #   --We could also try projecting some words from these new neighborhoods (maybe the top three from each?) onto our basis (just dot product) and getting those edges as well.
    e_dict = {}
    for i in range(3):
        e_dict[coord_map[i] + '_e'] = sum([[main_coords[i], coords[w][i], None] for w in id_to_name if w != main_word], [])
        e_dict[coord_map[i] + '_n'] = [coords[w][i] for w in id_to_name]

    return e_dict

def plot_network(g_dict, axis_titles):
    trace1=Scatter3d(
        x=g_dict['x_e'], y=g_dict['y_e'], z=g_dict['z_e'], mode='lines', line=Line(
                    color='rgb(125,125,125)', width=1), hoverinfo='none')

    trace2=Scatter3d(x=g_dict['x_n'], y=g_dict['y_n'], z=g_dict['z_n'], mode='markers', marker=Marker(
        symbol='dot', size=6, color='rgb(175,175,175)', line = Line(
            color='rgb(50,50,50)', width=0.5)), text=id_to_name, hoverinfo='text')

    #print(axis_titles)
    axes = [dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title=' ') for i in range(3)]
 
    title = 'word relationships with respect to ' + main_word
    layout=Layout(title=title,width=1000, height=1000, showlegend=False, scene=Scene(
        xaxis=XAxis(axes[0]), yaxis=YAxis(axes[1]), zaxis=ZAxis(axes[2])), margin=Margin(t=100), hovermode='closest')
    
    data = Data([trace1, trace2])
    fig=Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='arms_network.html')
    
def get_closest_basis_words(basis, model):
    maps = {}
    for i in range(basis.shape[0]):
        curr_bas = basis[i,:]
        closest = ('', 10)
        for key in model.wv.vocab:
#            if len(key) < 3:
#                continue
            word = key
            vec = model.wv[key]
            dist = abs(cosine_distance(vec, curr_bas))
            if dist < closest[1]:
                closest = (word, dist)
        maps[i] = closest
    return maps
                

if __name__ == '__main__':
    main_word = 'arms'
    model_path = 'second_amendment.bin'

    basis, coords, model = get_points_from_word_and_model(main_word, model_path, verbose=False, bigger_than=2)

    #basis isn't going to be used yet, but we might label the axes in the future
    #generate node-id mappings
    name_to_id = {}
    id_to_name = []
    temp_id = 0
    for w in coords:
        name_to_id[w] = temp_id
        temp_id +=1
        id_to_name.append(w)
    main_coords = coords[main_word] 
    coord_map= ['x','y','z']
    
    #generate network
    g_dict = generate_network(main_word, coords)
    
    #plot network
    plot_network(g_dict, get_closest_basis_words(basis, model))
