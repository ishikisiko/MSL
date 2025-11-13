# Model Compression and Optimization Pipeline

## Setup and Execution Instructions

### 1. Environment Setup
It is recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 2. Install Dependencies
Install the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 3. Running the Pipeline
The main execution pipeline is orchestrated in the `main.py` (or equivalent) script. This script will perform the following steps:
1.  Prepare the CIFAR-100 dataset.
2.  Train the baseline model (if not already trained).
3.  Run all compression experiments (pruning, quantization, distillation).
4.  Analyze the interactions between compression techniques.
5.  Evaluate all compressed models.
6.  Generate analysis reports, visualizations, and guidelines.

To run the full pipeline, execute:
```bash
python main.py 
```
*Note: You will need to create a `main.py` to orchestrate the calls to the different modules as outlined in `ASS4.md`.*

#### Quick Run on Google Colab
When prototyping in Colab, prefix shell commands with `!` (or `%cd` for directory changes). A minimal end-to-end session looks like:
```bash
!git clone <repo-url> MLS4
%cd MLS4
!pip install -r requirements.txt
!python main.py --train-baseline
```
Adjust the repo URL as needed and keep the `!` prefix to avoid Colab syntax errors.

### 4. Expected Output
- **Models**: The trained baseline model (`baseline_model.keras`) and all compressed model variants will be saved in the `compressed_models/` directory.
- **Checkpoints**: Training checkpoints will be saved in the `checkpoints/` directory.
- **Analysis Documents**: PDF reports, Markdown guidelines, and Excel benchmarks will be generated.
- **Visualizations**: PNG images of plots and heatmaps will be created.
