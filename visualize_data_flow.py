import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Add nodes for each dataset/table
G.add_node("Air Quality\nData", shape="rectangle")
G.add_node("Road Condition\nData", shape="rectangle")
G.add_node("Crime\nData", shape="rectangle")
G.add_node("Safety Index\nData", shape="rectangle")
G.add_node("Master\nDataset", shape="rectangle")
G.add_node("Risk Score\nPrediction", shape="rectangle")

# Add edges to show relationships
G.add_edge("Air Quality\nData", "Master\nDataset")
G.add_edge("Road Condition\nData", "Master\nDataset")
G.add_edge("Crime\nData", "Master\nDataset")
G.add_edge("Safety Index\nData", "Master\nDataset")
G.add_edge("Master\nDataset", "Risk Score\nPrediction")

# Set up the plot
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)

# Draw the graph
nx.draw(G, pos, 
        with_labels=True,
        node_color='lightblue',
        node_size=3000,
        font_size=10,
        font_weight='bold',
        arrows=True,
        edge_color='gray',
        arrowsize=20)

plt.title("TripGuard-AI Data Flow Diagram", pad=20)
plt.axis('off')

# Save the plot
plt.savefig('data_flow_diagram.png', bbox_inches='tight', dpi=300)
plt.close()