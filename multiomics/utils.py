import re
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gseapy as gp
from gseapy import barplot, dotplot
import networkx as nx 
from matplotlib import colormaps

def get_colors(groups):
    # Using a Seaborn palette
    unique_groups = list(set(groups))  # Unique list of groups
    palette = sns.color_palette('tab10', len(unique_groups))  # Replace "hsv" with any palette you prefer

    # Map groups to colors
    group_to_color = {group: color for group, color in zip(unique_groups, palette)}

    # Get node colors based on their group, aligning with the order of 'nodes'
    node_colors = [group_to_color[group] for group in groups]

    return node_colors, group_to_color

class CircInteractome():
    def __init__(self, filepath):
        self.annotation = pd.read_csv(filepath, sep='\t')

    def arraystar2circbase(self, arraystar_id):
        return self.annotation.set_index('circRNA').loc[arraystar_id].get('circbaseID')
        
    def RBProteins(self, circ_rna_id):
        url = f'https://circinteractome.nia.nih.gov/api/v2/circsearch?circular_rna_query={circ_rna_id}'
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text,features='lxml')
        search_genes = re.compile("data.setCell\(\d+, 0, '(.*)'\);")
        return search_genes.findall(soup.find_all('script')[2].text)

    def miRNA(self, circ_rna_id):
        url = f'https://circinteractome.nia.nih.gov/api/v2/mirnasearch?circular_rna_query={circ_rna_id}&mirna_query=&submit=miRNA+Target+Search'

        resp = requests.get(url)
        soup = BeautifulSoup(resp.text,features='lxml')
        search_mirna = re.compile("hsa-miR-[^']+")
        return search_mirna.findall(soup.find_all('script')[2].text)

class StringDB():
    def __init__(self, protein_list=None, species='9606' ):
        self.protein_list = protein_list
        self.species = species
        self.interactions = None
    
    def query(self, protein_list=None, species='9606'):

        if protein_list:
            self.protein_list = protein_list
            self.species = species

        proteins = '%0d'.join(self.protein_list)
        url = f'https://string-db.org/api/tsv/network?identifiers={proteins}&species={species}'
        resp = requests.get(url)
        if not resp.ok:
            return None

        lines = resp.text.split('\n') # pull the text from the response object and split based on new lines

        data = [l.split('\t') for l in lines] # split each line into its components based on tabs
        # convert to dataframe using the first row as the column names; drop empty, final row
        df = pd.DataFrame(data[1:-1], columns = data[0]) 
        # dataframe with the preferred names of the two proteins and the score of the interaction
        interactions = df[['preferredName_A', 'preferredName_B', 'score']] 


        G=nx.Graph(name='Protein Interaction Graph')
        interactions = np.array(interactions)
        for i in range(len(interactions)):
            interaction = interactions[i]
            a = interaction[0] # protein a node
            b = interaction[1] # protein b node
            w = float(interaction[2]) # score as weighted edge where high scores = low weight
            G.add_weighted_edges_from([(a,b,w)]) # add weighted edge to graph

        self.interactions = G
        return G

    def plot_interactions(self, colormap='plasma'):

        # function to rescale list of values to range [newmin,newmax]
        def rescale(l,newmin,newmax):
            arr = list(l)
            return [(x-min(arr))/(max(arr)-min(arr))*(newmax-newmin)+newmin for x in arr]

        G = self.interactions

        graph_colormap = colormaps.get_cmap(colormap)
        # node color varies with Degree
        c = rescale([G.degree(v) for v in G],0.0,0.9) 
        c = [graph_colormap(i) for i in c]
        # node size varies with betweeness centrality - map to range [10,100] 
        bc = nx.betweenness_centrality(G) # betweeness centrality
        s =  rescale([v for v in bc.values()],1500,7000)
        # edge width shows 1-weight to convert cost back to strength of interaction 
        ew = rescale([float(G[u][v]['weight']) for u,v in G.edges],0.1,4)
        # edge color also shows weight
        ec = rescale([float(G[u][v]['weight']) for u,v in G.edges],0.1,1)
        ec = [graph_colormap(i) for i in ec]

        pos = nx.spring_layout(G)
        plt.figure(figsize=(19,9),facecolor=[0.7,0.7,0.7,0.4])
        nx.draw_networkx(G, pos=pos, 
                        with_labels=True, 
                        node_color=c, 
                        node_size=s,
                        edge_color= ec,
                        width=ew,
                        font_color='white',
                        font_weight='bold',
                        font_size=11)
        plt.axis('off')
        plt.show()
    
def plot_GO(gene_list, gene_sets=['GO_Molecular_Function_2023']):
    # if you are only intrested in dataframe that enrichr returned, please set outdir=None
    enr = gp.enrichr(gene_list=gene_list, # or "./tests/data/gene_list.txt",
                    gene_sets=gene_sets,
                    organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                    outdir=None, # don't write to disk
                    )

    # to save your figure, make sure that ``ofname`` is not None
    ax = dotplot(enr.res2d, 
                title='Gene set enrichment',
                cmap='viridis_r', 
                size=10, 
                figsize=(3,5))

    return enr
    
# def draw_boxplot(data):
#     # Create a boxplot without whiskers
#     ax = sns.boxplot(data=data, x='Group', y='Level', whis=0, hue='Group', legend='brief')
#     # Remove the x-axis label
#     ax.set_xlabel('')

#     ax = sns.swarmplot(data=data, x='Group', y='Level', c='k')
#     # Remove the x-axis label
#     ax.set_xlabel('')


#     # Calculate and add whiskers for each group
#     for group in data['Group'].unique():
#         group_data = data[data['Group'] == group]

#         Q1 = np.percentile(group_data['Level'], 25)
#         Q3 = np.percentile(group_data['Level'], 75)
#         IQR = Q3 - Q1
#         mean = np.mean(group_data['Level'])

#         lower_whisker = mean - 1.5 * IQR
#         upper_whisker = mean + 1.5 * IQR

#         # Determine the group's position on the x-axis
#         group_position = np.where(data['Group'].unique() == group)[0][0]

#         # Add whisker lines
#         plt.plot([group_position, group_position], [lower_whisker, Q1], color='black', linewidth=1.0)
#         plt.plot([group_position, group_position], [upper_whisker, Q3], color='black', linewidth=1.0)

#         # Optionally, add horizontal lines at the ends of the whiskers
#         plt.plot([group_position - 0.1, group_position + 0.1], [lower_whisker, lower_whisker], color='black', linewidth=1.0)
#         plt.plot([group_position - 0.1, group_position + 0.1], [upper_whisker, upper_whisker], color='black', linewidth=1.0)

def draw_boxplot(data):
    # Create a boxplot without whiskers
    ax = sns.boxplot(data=data, x='Group', y='Level', whis=0, hue='Group', legend='brief')
    # Remove the x-axis label
    ax.set_xlabel('')

    ax = sns.swarmplot(data=data, x='Group', y='Level', c='k')
    # Remove the x-axis label
    ax.set_xlabel('')
    # Calculate and add whiskers for each group
    for group in data['Group'].unique():
        group_data = data[data['Group'] == group]

        Q1 = np.percentile(group_data['Level'], 25)
        Q3 = np.percentile(group_data['Level'], 75)
        IQR = Q3 - Q1
        mean = np.mean(group_data['Level'])

        lower_whisker = mean - 1.5 * IQR
        upper_whisker = mean + 1.5 * IQR

        # Determine the group's position on the x-axis
        group_position = np.where(data['Group'].unique() == group)[0][0]

        # Add whisker lines
        plt.plot([group_position, group_position], [lower_whisker, Q1], color='black', linewidth=1.0)
        plt.plot([group_position, group_position], [upper_whisker, Q3], color='black', linewidth=1.0)

        # Optionally, add horizontal lines at the ends of the whiskers
        plt.plot([group_position - 0.1, group_position + 0.1], [lower_whisker, lower_whisker], color='black', linewidth=1.0)
        plt.plot([group_position - 0.1, group_position + 0.1], [upper_whisker, upper_whisker], color='black', linewidth=1.0)
               
def draw_boxplot_2(data):
    # Create a boxplot without whiskers
    sns.boxplot(data=data, whis=0, showfliers=0, legend='brief')
    sns.stripplot(data=data, color='k')
    # Calculate and add whiskers for each group
    for group_position, group in enumerate(data):
        group_data = data[group]

        Q1 = np.percentile(group_data, 25)
        Q3 = np.percentile(group_data, 75)
        IQR = Q3 - Q1
        mean = np.mean(group_data)

        lower_whisker = mean - 1.5 * IQR
        upper_whisker = mean + 1.5 * IQR

        # Determine the group's position on the x-axis
        # group_position = np.where(data['Group'].unique() == group)[0][0]

        # Add whisker lines
        plt.plot([group_position, group_position], [lower_whisker, Q1], color='black')
        plt.plot([group_position, group_position], [upper_whisker, Q3], color='black')

        # Optionally, add horizontal lines at the ends of the whiskers
        plt.plot([group_position - 0.1, group_position + 0.1], [lower_whisker, lower_whisker], color='black')
        plt.plot([group_position - 0.1, group_position + 0.1], [upper_whisker, upper_whisker], color='black')
    
import numpy as N
import pylab as P

def _blob(x,y,area,colour):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = N.sqrt(area) / 2
    xcorners = N.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = N.array([y - hs, y - hs, y + hs, y + hs])
    P.fill(xcorners, ycorners, colour, edgecolor=colour)

def hinton(W, maxWeight=None):
    """
    Draws a Hinton diagram for visualizing a weight matrix. 
    Temporarily disables matplotlib interactive mode if it is on, 
    otherwise this takes forever.
    """
    reenable = False
    if P.isinteractive():
        P.ioff()
    P.clf()
    height, width = W.shape
    if not maxWeight:
        maxWeight = 2**N.ceil(N.log(N.max(N.abs(W)))/N.log(2))

    P.fill(N.array([0,width,width,0]),N.array([0,0,height,height]),'gray')
    P.axis('off')
    P.axis('equal')
    for x in xrange(width):
        for y in xrange(height):
            _x = x+1
            _y = y+1
            w = W[y,x]
            if w > 0:
                _blob(_x - 0.5, height - _y + 0.5, min(1,w/maxWeight),'white')
            elif w < 0:
                _blob(_x - 0.5, height - _y + 0.5, min(1,-w/maxWeight),'black')
    if reenable:
        P.ion()
    P.show()