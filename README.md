# 🤖 AI Chatbot System

[![Python Version](https://img.shields.io/badge/python-3.11.9-blue)](https://www.python.org/downloads/release/python-3119/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

## 📖 Table of Contents
- [🌟 Description](#description)
- [🛠️ Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Create and Activate Virtual Environment](#create-and-activate-virtual-environment)
  - [Install Dependencies](#install-dependencies)
- [🚀 Usage](#usage)
- [📄 License](#license)

<a id="description"></a>
## 🌟 Description
An intelligent chatbot system featuring natural language processing and machine learning capabilities. The chatbot understands user intents, analyzes sentiment, and maintains conversation context for more human-like interactions.

<a id="setup"></a>
## 🛠️ Setup

<a id="prerequisites"></a>
### Prerequisites
- Python 3.11.9
- pip package manager
- Recommended: 4GB+ RAM for training

<a id="create-and-activate-virtual-environment"></a>
### Create and Activate Virtual Environment

```bash
# Create a virtual environment
python -m venv .venv
```

```bash
# activate virtual environment
.venv\Scripts\activate
```

<a id="install-dependencies"></a>
### Install Dependencies

```bash
# install the dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

<a id="usage"></a>
## 🚀 Usage

```bash
# Train the Model
python train.py
```

```bash
# Run the Chatbot
python app.py
```

<a id="license"></a>
## 📄 License

[MIT](https://choosealicense.com/licenses/mit/)
