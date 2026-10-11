import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import streamlit as st

def plot_model_architecture(model_type, layers):
    """Visualize neural network architecture"""
    G = nx.DiGraph()
    
    # Add nodes for each layer
    for i, layer in enumerate(layers):
        G.add_node(f"Layer {i+1}\n{layer['type']}\n({layer['size']})", 
                  pos=(i, 0), layer_type=layer['type'])
    
    # Add edges between layers
    for i in range(len(layers)-1):
        G.add_edge(f"Layer {i+1}\n{layers[i]['type']}\n({layers[i]['size']})", 
                  f"Layer {i+2}\n{layers[i+1]['type']}\n({layers[i+1]['size']})")
    
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(12, 6))
    
    # Color nodes by layer type
    node_colors = []
    for node in G.nodes():
        layer_type = G.nodes[node]['layer_type']
        if 'LSTM' in layer_type or 'RNN' in layer_type:
            node_colors.append('#ff7f0e')
        elif 'Conv' in layer_type or 'CNN' in layer_type:
            node_colors.append('#2ca02c')
        elif 'Dense' in layer_type or 'Linear' in layer_type:
            node_colors.append('#d62728')
        elif 'Attention' in layer_type or 'Transformer' in layer_type:
            node_colors.append('#9467bd')
        else:
            node_colors.append('#1f77b4')
    
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color=node_colors,
            font_size=10, font_weight='bold', arrowsize=20, edge_color='gray')
    
    plt.title(f"{model_type} Architecture", fontsize=16, fontweight='bold')
    st.pyplot(plt)
    plt.close()

def plot_training_history(history):
    """Plot training metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Training Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title('Loss Over Epochs', fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    if 'train_acc' in history and 'val_acc' in history:
        ax2.plot(history['train_acc'], label='Training Accuracy', linewidth=2)
        ax2.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
        ax2.set_title('Accuracy Over Epochs', fontweight='bold')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_confusion_matrix(cm, classes):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix', fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    st.pyplot(plt)
    plt.close()