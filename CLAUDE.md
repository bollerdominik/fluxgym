# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FluxGym is a web UI for training FLUX LoRA models with low VRAM support (12GB/16GB/20GB). It combines the simplicity of AI-Toolkit's Gradio UI with the flexibility of Kohya Scripts for training.

**Architecture:**
- **Frontend**: Gradio web UI (`app.py`) forked from AI-Toolkit
- **Backend**: Training powered by Kohya Scripts (`sd-scripts/` submodule)
- **Models**: Supports multiple FLUX variants (dev, schnell, dev2pro) configured in `models.yaml`

## Key Components

### Core Files
- `app.py`: Main Gradio application with training UI and workflow orchestration
- `models.yaml`: Configuration for supported base models (FLUX variants)
- `sd-scripts/`: Kohya Scripts submodule for actual training execution
- `requirements.txt`: Python dependencies for the main app
- `sd-scripts/requirements.txt`: Dependencies for the training backend

### Directory Structure
- `datasets/`: Training datasets with images and caption files
- `outputs/`: Generated LoRA models and training artifacts
- `models/`: Downloaded base models (CLIP, UNET, VAE)

## Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv env

# Activate (Windows)
env\Scripts\activate

# Activate (Linux)
source env/bin/activate

# Install dependencies
cd sd-scripts
pip install -r requirements.txt
cd ..
pip install -r requirements.txt

# Install PyTorch (standard CUDA 12.1)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch (RTX 50-series with CUDA 12.8)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -U bitsandbytes
```

### Running the Application
```bash
# Start the web UI (ensure venv is activated)
python app.py
```

### Docker Development
```bash
# Build and run with Docker Compose
docker compose up -d --build

# Access at http://localhost:7860
```

### Testing
The sd-scripts submodule includes tests in `sd-scripts/tests/`. Run with pytest:
```bash
cd sd-scripts
pytest tests/
```

## Code Architecture

### Training Pipeline
1. **Dataset Preparation**: Images and captions processed through `datasets/` directory
2. **Model Configuration**: Base models defined in `models.yaml` with download URLs
3. **Training Execution**: Kohya Scripts handle the actual LoRA training
4. **Output Generation**: Trained models saved to `outputs/` with metadata

### Key Integration Points
- `app.py` orchestrates the entire training workflow
- Gradio UI dynamically generates advanced options by parsing Kohya script flags
- Model downloading handled automatically based on `models.yaml` configuration
- Training parameters passed through to underlying Kohya Scripts

### Model Support
Models are configured in `models.yaml` with:
- Hugging Face repository references
- License information
- Base model specifications for inference

### Advanced Features
- **Sample Image Generation**: Automatic sample generation during training with configurable prompts and intervals
- **Hugging Face Publishing**: Direct upload of trained LoRAs to Hugging Face Hub
- **Advanced Training Options**: Full access to all Kohya Scripts parameters through expandable UI

## Important Notes

- The project requires NVIDIA GPU with sufficient VRAM
- Models are automatically downloaded when first used
- Training datasets follow specific naming conventions (image files + matching .txt caption files)
- The Kohya Scripts submodule should remain on the `sd3` branch
- Environment activation is required before running any commands
- NEVER WRITE ANY TESTS with AI