import os
from dataclasses import dataclass

@dataclass
class AppConfig:
    app_title: str = "Neural Network Explorer"
    app_description: str = "Interactive comparison of different neural network architectures"
    theme_primary_color: str = "#1f77b4"
    max_sequence_length: int = 100
    default_epochs: int = 10
    cache_dir: str = ".cache"
    
    def __post_init__(self):
        os.makedirs(self.cache_dir, exist_ok=True)