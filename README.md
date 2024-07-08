# Minsk: An MLOps Pipeline for MNIST

Minsk is a streamlined MLOps pipeline designed specifically for the MNIST dataset, one of the most iconic datasets in the machine learning community. This project aims to provide an end-to-end solution for training, evaluating, and deploying models that recognize handwritten digits. This is a solved problem, and more or less being created so I can play with the MLOps libraries/frameworks/software.

## Features

- **Data Preprocessing**: Normalization and one-hot encoding to prepare the MNIST dataset for training.
- **Model Training**: A Sequential model comprising a Flatten layer, followed by two Dense layers, trained on the MNIST dataset.
- **Evaluation and Testing**: Metrics for accuracy assessment and visualization of model predictions.
- **Model Deployment**: Instructions for containerizing the model using Docker for easy deployment.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.10 or later
- Docker (for deployment)

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/minsk.git
    ```

2. Install packages
    ```sh
    pip install -r requirements.txt
    ```

