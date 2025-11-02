import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Define node positions more explicitly
pos = {
    'Graph2': (0.5, 1),    # Top center
    'Time': (0, 0),        # Bottom left
    'City': (0.5, 0),      # Bottom center
    'Site': (1, 0)         # Bottom right
}

# Add nodes
G.add_nodes_from(['Graph2', 'Time', 'City', 'Site'])

# Add edges (relationships)
edges = [
    ('Time', 'City'),
    ('City', 'Site'),
    ('Graph2', 'City')
]
G.add_edges_from(edges)

# Set up the plot with a larger figure size and white background
plt.figure(figsize=(12, 8), facecolor='white')
plt.gca().set_facecolor('white')

# Draw nodes (boxes)
for node in G.nodes():
    plt.gca().add_patch(
        plt.Rectangle(
            (pos[node][0] - 0.1, pos[node][1] - 0.05),
            0.2, 0.1,
            facecolor='lightblue',
            edgecolor='black',
            alpha=0.7
        )
    )
    plt.text(
        pos[node][0], pos[node][1],
        node,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=14,
        fontweight='bold'
    )

# Draw edges with arrows
for edge in G.edges():
    start = pos[edge[0]]
    end = pos[edge[1]]
    plt.arrow(
        start[0], start[1],
        end[0] - start[0], end[1] - start[1],
        head_width=0.03,
        head_length=0.05,
        fc='black',
        ec='black',
        length_includes_head=True,
        alpha=0.6
    )

# Set the plot limits with some padding
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)

# Remove axes
plt.axis('off')

# Add title
plt.title('Data Relationship Diagram', pad=20, fontsize=16, fontweight='bold')

# Save with high resolution and tight layout
plt.savefig('data_relationship_graph.png', 
            bbox_inches='tight',
            dpi=300,
            facecolor='white',
            edgecolor='none')
plt.close()

print("Updated graph has been created and saved as 'data_relationship_graph.png'")