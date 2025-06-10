# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment

- Python 3.12.3 with virtual environment at `.venv/`
- Activate environment: `source .venv/bin/activate`
- Run main optimization pipeline: `python3 scripts/run_optimization.py`

## Core Architecture

VED_decomposer is a scientific computing application for electron density analysis using non-linear optimization techniques. The codebase follows a task-based architecture using Prefect for workflow orchestration.

### Key Components

**Main Pipeline Flow** (`src/flows/optimization_flow.py`):
- Entry point for the optimization pipeline using Prefect flows
- Coordinates pre-processing, data processing, and post-processing stages
- Currently switches between `main_loop` (Grassmann manifold optimization) and `main_loop_PSD` (PSD optimization with convex methods)

**Data Processing Approaches**:
1. **Grassmann Manifold Optimization** (`src/tasks/data_processing/main_loop.py`):
   - Uses pymanopt for manifold optimization on Grassmann spaces
   - Employs autograd for automatic differentiation
   - Multi-stage magnification approach (4x, 2x, 1x)

2. **PSD Optimization** (`src/tasks/data_processing/main_loop_PSD.py`):
   - Uses cvxpy for convex optimization with PSD constraints
   - Falls back to scipy.optimize for non-convex problems
   - Automatically detects autograd availability

**Orbital Generation** (`src/tasks/pre_processing/create_orbitals.py`):
- Creates atomic orbitals for different magnification levels
- Handles 3d, 4s, 4p orbitals based on settings configuration

**Settings Management** (`src/tasks/pre_processing/settings.py`):
- YAML-based configuration system
- Defines grid parameters, orbital types, and optimization settings
- Sample configuration at `data/input_sample/settings.yaml`

### Data Flow

1. Load configuration from YAML file (gridSize, orbital types, center coordinates)
2. Import experimental data from .xplor format
3. Generate atomic orbitals at specified magnification levels
4. Run optimization pipeline (either Grassmann or PSD approach)
5. Output results to `output/` directory in various formats (.xplor, .txt)

### Key Dependencies

- **pymanopt**: Manifold optimization on Grassmann spaces
- **cvxpy**: Convex optimization for PSD constraints  
- **autograd**: Automatic differentiation (optional, falls back to numerical)
- **prefect**: Workflow orchestration
- **scipy**: Scientific computing utilities
- **numpy**: Numerical arrays and operations

## Testing

Run tests with: `python3 -m unittest discover tests/`

Note: pytest is not available in this environment; use unittest instead.