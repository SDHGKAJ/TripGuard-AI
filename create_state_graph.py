import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def create_state_graph(state_name):
    # Load and prepare data
    try:
        # Load crime data
        crime_df = pd.read_csv('crime_dataset_india.csv')
        crime_df['Date Reported'] = pd.to_datetime(crime_df['Date Reported'], format='%d-%m-%Y %H:%M')
        
        # Filter data for the specified state/city
        state_data = crime_df[crime_df['City'].str.upper() == state_name.upper()]
        
        if len(state_data) == 0:
            print(f"No data found for {state_name}")
            return
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Calculate statistics
        total_crimes = len(state_data)
        crime_types = state_data['Crime Description'].value_counts()
        most_common_crime = crime_types.index[0] if len(crime_types) > 0 else "No data"
        time_distribution = state_data['Time of Occurrence'].value_counts()
        
        # Create node labels with data
        time_label = f"Time Analysis\nTotal Records: {total_crimes}"
        city_label = f"{state_name}\nMost Common: {most_common_crime}"
        stats_label = f"Statistics\nCrime Types: {len(crime_types)}"
        
        # Define node positions
        pos = {
            'Graph2': (0.5, 1),    # Top center
            'Time': (0, 0),        # Bottom left
            'City': (0.5, 0),      # Bottom center
            'Stats': (1, 0)        # Bottom right
        }
        
        # Set up the plot
        plt.figure(figsize=(12, 8), facecolor='white')
        plt.gca().set_facecolor('white')
        
        # Draw nodes (boxes)
        node_labels = {
            'Graph2': 'Crime Analysis',
            'Time': time_label,
            'City': city_label,
            'Stats': stats_label
        }
        
        # Add nodes and edges
        for node, label in node_labels.items():
            G.add_node(node)
            plt.gca().add_patch(
                plt.Rectangle(
                    (pos[node][0] - 0.15, pos[node][1] - 0.08),
                    0.3, 0.16,  # Made boxes larger
                    facecolor='lightblue',
                    edgecolor='black',
                    alpha=0.7,
                    linewidth=2
                )
            )
            # Add multiline text
            plt.text(
                pos[node][0], pos[node][1],
                label,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=10,
                fontweight='bold',
                wrap=True
            )
        
        # Add edges
        edges = [
            ('Time', 'City'),
            ('City', 'Stats'),
            ('Graph2', 'City')
        ]
        
        # Draw edges with arrows
        for edge in edges:
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
        
        # Set the plot limits with padding
        plt.xlim(-0.3, 1.3)
        plt.ylim(-0.2, 1.2)
        
        # Remove axes
        plt.axis('off')
        
        # Add title with state name
        plt.title(f'Crime Data Relationship - {state_name}', pad=20, fontsize=16, fontweight='bold')
        
        # Save the visualization
        output_file = f'crime_graph_{state_name.lower().replace(" ", "_")}.png'
        plt.savefig(output_file, bbox_inches='tight', dpi=300, facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Graph has been created and saved as '{output_file}'")
        
        # Print additional statistics
        print(f"\nStatistics for {state_name}:")
        print(f"Total crimes recorded: {total_crimes}")
        print("\nTop 5 most common crimes:")
        print(crime_types.head().to_string())
        
    except Exception as e:
        print(f"Error creating graph: {str(e)}")

# Function to get list of available cities
def list_available_cities():
    try:
        crime_df = pd.read_csv('crime_dataset_india.csv')
        cities = sorted(crime_df['City'].unique())
        print("\nAvailable cities:")
        for city in cities:
            print(f"- {city}")
    except Exception as e:
        print(f"Error listing cities: {str(e)}")

if __name__ == "__main__":
    # Example usage
    print("Crime Data Visualization Tool")
    print("----------------------------")
    list_available_cities()
    
    state_name = input("\nEnter city name (e.g., Chennai, Mumbai, Delhi): ")
    if state_name:
        create_state_graph(state_name)