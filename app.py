import streamlit as st
import numpy as np
import torch
import time
import json
from config import AppConfig
from utils.visualization import plot_model_architecture, plot_training_history, plot_confusion_matrix
from models.rnn_model import SimpleRNN, RNNDataset, train_rnn_model
from models.lstm_model import LSTMModel, LSTMDataset, train_lstm_model
from models.cnn_model import CNNModel, CNNDataset, train_cnn_model
from models.slm_model import SmallLanguageModel, SLMDataset
from models.llm_interface import LLMComparison

# Set page configuration
st.set_page_config(
    page_title="Neural Network Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
config = AppConfig()

# Custom CSS for professional styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #2c3e50;
        color: white;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .model-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.title("🧠 Neural Network Explorer")
    st.markdown("### Interactive comparison of different neural network architectures")
    st.markdown("---")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Select Page", [
            "🏠 Home",
            "🔄 RNN Demo",
            "🌊 LSTM Demo", 
            "🖼️ CNN Demo",
            "💬 SLM Demo",
            "🤖 LLM Demo",
            "📊 Comparison",
            "ℹ️ About"
        ])
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔄 RNN Demo":
        show_rnn_demo()
    elif page == "🌊 LSTM Demo":
        show_lstm_demo()
    elif page == "🖼️ CNN Demo":
        show_cnn_demo()
    elif page == "💬 SLM Demo":
        show_slm_demo()
    elif page == "🤖 LLM Demo":
        show_llm_demo()
    elif page == "📊 Comparison":
        show_comparison()
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page():
    st.header("Welcome to Neural Network Explorer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Project Features
        
        - **Interactive Demos**: Hands-on experience with different neural network architectures
        - **Real-time Visualization**: See model architectures and training progress
        - **Performance Comparison**: Compare different models on various tasks
        - **Educational Content**: Learn about each architecture's strengths and weaknesses
        - **Professional UI**: Clean, intuitive interface built with Streamlit [[1]]
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Supported Architectures
        
        - **RNN (Recurrent Neural Network)**: Sequential data processing
        - **LSTM (Long Short-Term Memory)**: Advanced sequence modeling with memory
        - **CNN (Convolutional Neural Network)**: Image recognition and computer vision
        - **SLM (Small Language Model)**: Efficient text processing
        - **LLM (Large Language Model)**: Advanced natural language understanding
        """)
    
    st.markdown("---")
    
    # Quick demo section
    st.subheader("Quick Demo")
    st.markdown("Select a model type from the sidebar to get started!")
    
    # System info
    st.markdown("---")
    st.subheader("System Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        device = "GPU" if torch.cuda.is_available() else "CPU"
        st.metric("Device", device)
    
    with col2:
        st.metric("PyTorch Version", torch.__version__)
    
    with col3:
        st.metric("Streamlit Version", st.__version__)

def show_rnn_demo():
    st.header("🔄 Recurrent Neural Network (RNN) Demo")
    
    # Model architecture visualization
    st.subheader("Model Architecture")
    rnn_layers = [
        {'type': 'Input', 'size': 'Sequence Length × Features'},
        {'type': 'RNN Layer', 'size': '128 units'},
        {'type': 'RNN Layer', 'size': '64 units'}, 
        {'type': 'Output', 'size': 'Classes'}
    ]
    plot_model_architecture("RNN", rnn_layers)
    
    st.markdown("""
    **RNN Characteristics:**
    - Processes sequential data one step at a time
    - Maintains hidden state that captures information about previous steps
    - Suffers from vanishing/exploding gradients for long sequences
    - Good for short-term dependencies
    
    **Use Cases:** Time series prediction, simple sequence classification, basic text processing
    """)
    
    # Hyperparameter controls
    st.subheader("Hyperparameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hidden_size = st.slider("Hidden Size", 32, 256, 128, 32)
    with col2:
        num_layers = st.slider("Number of Layers", 1, 4, 2, 1)
    with col3:
        learning_rate = st.slider("Learning Rate", 0.0001, 0.1, 0.001, 0.0001, format="%.4f")
    
    # Training section
    if st.button("🚀 Train RNN Model"):
        with st.spinner("Training RNN model..."):
            # Create dummy data for demonstration
            seq_length = 20
            input_size = 10
            num_classes = 3
            batch_size = 32
            
            # Generate synthetic data
            X_train = np.random.randn(1000, seq_length, input_size)
            y_train = np.random.randint(0, num_classes, 1000)
            X_val = np.random.randn(200, seq_length, input_size)
            y_val = np.random.randint(0, num_classes, 200)
            
            # Create datasets and dataloaders
            train_dataset = RNNDataset(X_train, y_train)
            val_dataset = RNNDataset(X_val, y_val)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Initialize and train model
            model = SimpleRNN(input_size, hidden_size, num_classes, num_layers)
            trained_model, history = train_rnn_model(model, train_loader, val_loader, 
                                                    epochs=10, lr=learning_rate)
            
            # Show results
            st.success("✅ Training completed!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Training History")
                plot_training_history(history)
            
            with col2:
                st.subheader("Performance Metrics")
                final_train_acc = history['train_acc'][-1]
                final_val_acc = history['val_acc'][-1]
                
                st.metric("Final Training Accuracy", f"{final_train_acc:.2f}%")
                st.metric("Final Validation Accuracy", f"{final_val_acc:.2f}%")
                
                # Confusion matrix (dummy data for demo)
                dummy_cm = np.random.randint(5, 20, (num_classes, num_classes))
                plot_confusion_matrix(dummy_cm, [f"Class {i}" for i in range(num_classes)])

def show_lstm_demo():
    st.header("🌊 Long Short-Term Memory (LSTM) Demo")
    
    # Model architecture visualization
    st.subheader("Model Architecture")
    lstm_layers = [
        {'type': 'Input', 'size': 'Sequence Length × Features'},
        {'type': 'LSTM Layer', 'size': '256 units'},
        {'type': 'LSTM Layer', 'size': '128 units'},
        {'type': 'Dropout', 'size': '0.2'},
        {'type': 'Output', 'size': 'Classes'}
    ]
    plot_model_architecture("LSTM", lstm_layers)
    
    st.markdown("""
    **LSTM Advantages over RNN:**
    - Uses gating mechanisms (input, forget, output gates) to control information flow
    - Can learn long-term dependencies effectively
    - Less susceptible to vanishing gradient problem
    - Better at capturing complex patterns in sequential data
    
    **Use Cases:** Speech recognition, machine translation, sentiment analysis, time series forecasting [[4]]
    """)
    
    # Hyperparameter controls
    st.subheader("Hyperparameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hidden_size = st.slider("Hidden Size", 64, 512, 256, 64)
    with col2:
        num_layers = st.slider("Number of Layers", 1, 4, 2, 1)
    with col3:
        dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.1)
    
    # Training section
    if st.button("🚀 Train LSTM Model"):
        with st.spinner("Training LSTM model... This may take a few minutes"):
            # Create dummy data for demonstration
            seq_length = 30
            input_size = 15
            num_classes = 5
            batch_size = 64
            
            # Generate synthetic data
            X_train = np.random.randn(1500, seq_length, input_size)
            y_train = np.random.randint(0, num_classes, 1500)
            X_val = np.random.randn(300, seq_length, input_size)
            y_val = np.random.randint(0, num_classes, 300)
            
            # Create datasets and dataloaders
            train_dataset = LSTMDataset(X_train, y_train)
            val_dataset = LSTMDataset(X_val, y_val)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Initialize and train model
            model = LSTMModel(input_size, hidden_size, num_classes, num_layers, dropout)
            trained_model, history = train_lstm_model(model, train_loader, val_loader, 
                                                    epochs=15, lr=0.001)
            
            # Show results
            st.success("✅ LSTM training completed successfully!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Training Progress")
                plot_training_history(history)
            
            with col2:
                st.subheader("Model Performance")
                final_train_acc = history['train_acc'][-1]
                final_val_acc = history['val_acc'][-1]
                
                st.metric("Training Accuracy", f"{final_train_acc:.2f}%")
                st.metric("Validation Accuracy", f"{final_val_acc:.2f}%")
                
                # Show gating mechanism visualization
                st.markdown("### LSTM Gating Mechanism")
                st.image("https://miro.medium.com/v2/resize:fit:1400/1*XISs5JxAJyz66Z7LZ1D7wA.gif", 
                        caption="LSTM gates control information flow", width=400)

def show_cnn_demo():
    st.header("🖼️ Convolutional Neural Network (CNN) Demo")
    
    # Model architecture visualization
    st.subheader("Model Architecture")
    cnn_layers = [
        {'type': 'Input', 'size': '28×28×1'},
        {'type': 'Conv2D', 'size': '32 filters, 3×3'},
        {'type': 'ReLU + MaxPool', 'size': '2×2'},
        {'type': 'Conv2D', 'size': '64 filters, 3×3'},
        {'type': 'ReLU + MaxPool', 'size': '2×2'},
        {'type': 'Conv2D', 'size': '128 filters, 3×3'},
        {'type': 'ReLU + MaxPool', 'size': '2×2'},
        {'type': 'Flatten', 'size': '1152 units'},
        {'type': 'Dense', 'size': '512 units'},
        {'type': 'Dense', 'size': '256 units'},
        {'type': 'Output', 'size': '10 classes'}
    ]
    plot_model_architecture("CNN", cnn_layers)
    
    st.markdown("""
    **CNN Key Features:**
    - Uses convolutional layers to detect spatial patterns
    - Pooling layers reduce spatial dimensions and computational complexity
    - Translation invariant - can recognize patterns regardless of position
    - Hierarchical feature extraction (edges → textures → objects)
    
    **Use Cases:** Image classification, object detection, facial recognition, medical imaging [[6]]
    """)
    
    # Dataset selection
    st.subheader("Dataset Selection")
    dataset_choice = st.selectbox("Select Dataset", ["MNIST (Handwritten Digits)", "Fashion MNIST", "CIFAR-10 (Small Images)"])
    
    # Hyperparameter controls
    st.subheader("Model Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")
        batch_size = st.select_slider("Batch Size", [16, 32, 64, 128], 64)
    
    with col2:
        num_epochs = st.slider("Number of Epochs", 5, 50, 10, 5)
        data_augmentation = st.checkbox("Enable Data Augmentation", value=True)
    
    # Training section
    if st.button("🚀 Train CNN Model"):
        with st.spinner("Training CNN model... This may take several minutes"):
            # Create dummy image data for demonstration
            input_channels = 1  # Grayscale for MNIST-like data
            img_size = 28
            num_classes = 10
            
            # Generate synthetic image data
            X_train = np.random.randn(5000, input_channels, img_size, img_size)
            y_train = np.random.randint(0, num_classes, 5000)
            X_val = np.random.randn(1000, input_channels, img_size, img_size)
            y_val = np.random.randint(0, num_classes, 1000)
            
            # Create datasets and dataloaders
            train_dataset = CNNDataset(X_train, y_train)
            val_dataset = CNNDataset(X_val, y_val)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Initialize and train model
            model = CNNModel(input_channels, num_classes)
            trained_model, history = train_cnn_model(model, train_loader, val_loader, 
                                                   epochs=num_epochs, lr=learning_rate)
            
            # Show results
            st.success("✅ CNN training completed successfully!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Training Results")
                plot_training_history(history)
            
            with col2:
                st.subheader("Performance Analysis")
                final_train_acc = history['train_acc'][-1]
                final_val_acc = history['val_acc'][-1]
                
                st.metric("Training Accuracy", f"{final_train_acc:.2f}%")
                st.metric("Validation Accuracy", f"{final_val_acc:.2f}%")
                
                # Show feature maps
                st.markdown("### Feature Map Visualization")
                st.image("https://miro.medium.com/v2/resize:fit:1400/1*vkQ0hXDaQv57sALXAJquxA.jpeg", 
                        caption="CNN learns hierarchical features", width=400)

def show_slm_demo():
    st.header("💬 Small Language Model (SLM) Demo")
    
    # Model architecture visualization
    st.subheader("Model Architecture")
    slm_layers = [
        {'type': 'Input', 'size': 'Token IDs'},
        {'type': 'Embedding', 'size': '768 dimensions'},
        {'type': 'Transformer Block', 'size': '6 layers'},
        {'type': 'Pooling', 'size': 'Mean/CLS'},
        {'type': 'Classification', 'size': 'Classes'}
    ]
    plot_model_architecture("Small Language Model", slm_layers)
    
    st.markdown("""
    **SLM Characteristics:**
    - Distilled versions of large language models
    - Faster inference and lower resource requirements
    - Good balance between performance and efficiency
    - Suitable for edge deployment and real-time applications
    
    **Use Cases:** Text classification, sentiment analysis, chatbots, content moderation [[5]]
    """)
    
    # Task selection
    st.subheader("Task Selection")
    task_type = st.selectbox("Select Task", ["Sentiment Analysis", "Text Classification", "Named Entity Recognition"])
    
    # Model selection
    st.subheader("Model Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        model_choice = st.selectbox("Model Architecture", [
            "distilbert-base-uncased", 
            "albert-base-v2", 
            "mobilebert-uncased"
        ])
    
    with col2:
        max_length = st.slider("Max Sequence Length", 64, 512, 128, 64)
    
    # Interactive demo
    st.subheader("Interactive Demo")
    text_input = st.text_area("Enter text to analyze:", "I love this product! It works amazingly well.")
    
    if st.button("🎯 Analyze Text"):
        with st.spinner("Processing text with SLM..."):
            # Simulate SLM processing
            time.sleep(1)
            
            # Mock results based on input text
            if "love" in text_input.lower() or "amazing" in text_input.lower() or "great" in text_input.lower():
                sentiment = "Positive 😊"
                confidence = 0.95
            elif "hate" in text_input.lower() or "terrible" in text_input.lower() or "bad" in text_input.lower():
                sentiment = "Negative 😞"
                confidence = 0.89
            else:
                sentiment = "Neutral 😐"
                confidence = 0.75
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sentiment", sentiment)
            with col2:
                st.metric("Confidence", f"{confidence:.2%}")
            with col3:
                st.metric("Tokens", str(len(text_input.split())))
            
            # Show attention visualization
            st.subheader("Attention Visualization")
            st.image("https://miro.medium.com/v2/resize:fit:1400/1*9PJ4Xq72FpI4qjQqH9XhOA.png", 
                    caption="Model attention weights over text", width=600)

def show_llm_demo():
    st.header("🤖 Large Language Model (LLM) Demo")
    
    # Model comparison setup
    st.subheader("Available LLMs")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🟢 GPT-2 (Small)")
        st.markdown("- 124M parameters")
        st.markdown("- Fast inference")
        st.markdown("- Good for basic tasks")
    
    with col2:
        st.markdown("### 🔵 OPT-125M")
        st.markdown("- 125M parameters") 
        st.markdown("- Meta's open model")
        st.markdown("- Balanced performance")
    
    with col3:
        st.markdown("### 🟣 GPT-Neo-125M")
        st.markdown("- 125M parameters")
        st.markdown("- EleutherAI's model")
        st.markdown("- Strong text generation")
    
    st.markdown("---")
    
    # Interactive generation
    st.subheader("Text Generation Demo")
    
    # Model selection
    model_choice = st.selectbox("Select LLM", [
        "gpt2", 
        "facebook/opt-125m", 
        "EleutherAI/gpt-neo-125M"
    ])
    
    # Generation parameters
    col1, col2 = st.columns(2)
    
    with col1:
        prompt = st.text_input("Enter prompt:", "The future of artificial intelligence is")
        max_length = st.slider("Max Length", 50, 200, 100, 50)
    
    with col2:
        temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
        top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.1)
    
    if st.button("✨ Generate Text"):
        with st.spinner(f"Generating text with {model_choice}... This may take a moment"):
            try:
                # Initialize LLM comparison
                llm_comparison = LLMComparison([model_choice])
                
                # Generate text
                result = llm_comparison.models[model_choice].generate_text(
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p
                )
                
                # Display results
                st.success("✅ Generation completed!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Generated Text")
                    st.text_area("Result", result['generated_text'], height=300)
                
                with col2:
                    st.subheader("Generation Stats")
                    st.metric("Generation Time", f"{result['generation_time']:.2f} seconds")
                    st.metric("Token Count", result['token_count'])
                    st.metric("Tokens/second", f"{result['token_count']/result['generation_time']:.1f}")
                    
                    st.markdown("### Model Info")
                    model_info = llm_comparison.models[model_choice].get_model_info()
                    st.json({
                        'model_name': model_info['model_name'],
                        'device': model_info['device'],
                        'num_parameters': f"{model_info['num_parameters']:,}"
                    })
                    
            except Exception as e:
                st.error(f"Error generating text: {e}")
                st.error("This might be due to limited resources. Try a smaller model or shorter text.")

def show_comparison():
    st.header("📊 Neural Network Comparison")
    
    # Comparison table
    st.subheader("Architecture Comparison")
    
    comparison_data = {
        "Architecture": ["RNN", "LSTM", "CNN", "SLM", "LLM"],
        "Best For": [
            "Short sequences", 
            "Long sequences with memory", 
            "Spatial data (images)", 
            "Efficient text processing", 
            "Complex language tasks"
        ],
        "Training Speed": ["Fast", "Medium", "Fast", "Medium", "Very Slow"],
        "Memory Usage": ["Low", "Medium", "Medium", "Medium", "Very High"],
        "Typical Applications": [
            "Time series prediction",
            "Speech recognition, Translation",
            "Image classification, Object detection", 
            "Sentiment analysis, Chatbots",
            "Content generation, Complex reasoning"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df.style.set_properties(**{'text-align': 'left'}))
    
    st.markdown("---")
    
    # Performance benchmark
    st.subheader("Performance Benchmark")
    
    if st.button("🚀 Run Benchmark"):
        with st.spinner("Running benchmark comparison... This will take several minutes"):
            try:
                # Initialize LLM comparison for benchmarking
                llm_comparison = LLMComparison([
                    "gpt2",
                    "facebook/opt-125m", 
                    "EleutherAI/gpt-neo-125M"
                ])
                
                # Test prompts
                test_prompts = [
                    "The weather today is",
                    "Artificial intelligence will",
                    "In the future, humans will",
                    "The best way to learn is"
                ]
                
                # Run benchmark
                benchmark_results = llm_comparison.benchmark_performance(test_prompts, num_runs=2)
                
                # Display results
                st.success("✅ Benchmark completed!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Generation Speed")
                    speed_data = {model: results['avg_generation_time'] for model, results in benchmark_results.items()}
                    speed_df = pd.DataFrame({
                        'Model': list(speed_data.keys()),
                        'Avg Time (s)': list(speed_data.values())
                    })
                    st.bar_chart(speed_data)
                
                with col2:
                    st.subheader("Token Efficiency")
                    token_data = {model: results['avg_token_count'] for model, results in benchmark_results.items()}
                    st.bar_chart(token_data)
                
                # Detailed table
                st.subheader("Detailed Results")
                detailed_results = []
                for model_name, results in benchmark_results.items():
                    detailed_results.append({
                        'Model': model_name,
                        'Avg Generation Time (s)': f"{results['avg_generation_time']:.3f}",
                        'Avg Tokens Generated': f"{results['avg_token_count']:.1f}",
                        'Tokens/Second': f"{results['avg_token_count']/results['avg_generation_time']:.1f}"
                    })
                
                st.table(pd.DataFrame(detailed_results))
                
            except Exception as e:
                st.error(f"Benchmark error: {e}")
                st.error("This benchmark requires significant resources. Consider running on a more powerful machine.")

def show_about_page():
    st.header("ℹ️ About Neural Network Explorer")
    
    st.markdown("""
    ### 🎓 Project Purpose
    
    This project serves as an educational tool and demonstration platform for understanding different neural network architectures. It provides hands-on experience with:
    
    - **RNNs**: Understanding sequential data processing
    - **LSTMs**: Learning about memory mechanisms in neural networks  
    - **CNNs**: Exploring spatial feature extraction
    - **SLMs**: Working with efficient language models
    - **LLMs**: Experimenting with large-scale language understanding
    
    ### 🛠️ Technical Stack
    
    - **Frontend**: Streamlit for interactive UI [[1]]
    - **Backend**: PyTorch for deep learning models
    - **Language Models**: Hugging Face Transformers library
    - **Visualization**: Matplotlib, Seaborn, NetworkX
    - **Deployment**: Streamlit Cloud for easy sharing [[1]]
    
    ### 🚀 Best Practices Implemented
    
    - **Modular Architecture**: Clean separation of concerns for maintainability [[2]]
    - **Performance Optimization**: Efficient data loading and model training [[3]]
    - **Professional UI/UX**: Intuitive interface with clear visualizations
    - **Error Handling**: Robust error handling and user feedback
    - **Documentation**: Comprehensive comments and user guidance
    
    ### 🤝 Contributing
    
    This is an open educational project. Contributions are welcome! Please follow these guidelines:
    
    1. Fork the repository
    2. Create a feature branch
    3. Submit a pull request with detailed description
    4. Include tests and documentation
    
    ### 📧 Contact
    
    For questions, suggestions, or collaboration opportunities, please contact the project maintainers.
    
    ---
    
    **Note**: This is a demonstration project. The models are trained on synthetic data for educational purposes. For production use, you would need real datasets and more extensive training.
    """)

if __name__ == "__main__":
    main()