import uuid
import os

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter threading issues
import matplotlib.pyplot as plt


def create_line_graph(x_data, y_data, title="Line Graph", x_label="X", y_label="Y", save_dir="./graphs"):
    """
    Save a line graph using matplotlib.
    
    Parameters:
        x_data: List or array of x-axis values
        y_data: List or array of y-axis values
        title: Title of the graph
        x_label: Label for x-axis
        y_label: Label for y-axis
        save_dir: Directory to save the graph
    
    Returns:
        str: Path of the saved graph
    """
    # Generate unique filename using UUID
    # line_uuid = str(uuid.uuid4())
    line_uuid = '1'

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create the figure and plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    
    # Save the graph
    file_path = os.path.join(save_dir, f"line_graph_{line_uuid}.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return file_path

def create_bar_chart(x_data, y_data, title="Bar Chart", x_label="X", y_label="Y", save_dir="./graphs"):
    """
    Save a bar chart using matplotlib.
    
    Parameters:
        x_data: List or array of x-axis values (categories)
        y_data: List or array of y-axis values (heights)
        title: Title of the chart
        x_label: Label for x-axis
        y_label: Label for y-axis
        save_dir: Directory to save the chart
    
    Returns:
        str: Path of the saved chart
    """
    # Generate unique filename using UUID
    # bar_uuid = str(uuid.uuid4())
    bar_uuid = '1'
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create the figure and plot
    plt.figure(figsize=(10, 6))
    plt.bar(x_data, y_data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, axis='y')
    
    # Save the chart
    file_path = os.path.join(save_dir, f"bar_chart_{bar_uuid}.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return file_path


def create_pie_chart(labels, values, title="Pie Chart", save_dir="./graphs"):
    """
    Save a pie chart using matplotlib.
    
    Parameters:
        labels: List of category labels
        values: List of values for each category
        title: Title of the chart
        save_dir: Directory to save the chart
    
    Returns:
        str: Path of the saved chart
    """
    pie_uuid = '1'
    
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title(title)
    plt.axis('equal')
    
    file_path = os.path.join(save_dir, f"pie_chart_{pie_uuid}.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return file_path


def create_scatter_plot(x_data, y_data, title="Scatter Plot", x_label="X", y_label="Y", save_dir="./graphs"):
    """
    Save a scatter plot using matplotlib.
    
    Parameters:
        x_data: List or array of x-axis values
        y_data: List or array of y-axis values
        title: Title of the plot
        x_label: Label for x-axis
        y_label: Label for y-axis
        save_dir: Directory to save the plot
    
    Returns:
        str: Path of the saved plot
    """
    scatter_uuid = '1'
    
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, alpha=0.6, edgecolors='black', linewidth=0.5)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    
    file_path = os.path.join(save_dir, f"scatter_plot_{scatter_uuid}.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return file_path