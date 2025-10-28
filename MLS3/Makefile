# Hardware-Aware Design Assignment Makefile
# Supports both Track A (real hardware) and Track B (simulation)

.PHONY: help clean train test benchmark deploy simulate analyze

# Default target
help:
	@echo "Hardware-Aware Design Assignment Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  train       - Train baseline model"
	@echo "  optimize    - Run hardware-aware optimizations"
	@echo "  benchmark   - Benchmark all models"
	@echo "  deploy      - Deploy to target platforms (Track A)"
	@echo "  simulate    - Run simulations (Track B)"
	@echo "  analyze     - Generate analysis reports"
	@echo "  clean       - Clean generated files"
	@echo ""
	@echo "Usage examples:"
	@echo "  make train"
	@echo "  make PLATFORM=cpu_x86 deploy"
	@echo "  make SIMULATOR=qemu simulate"

# Configuration
PYTHON := python3
PART1 := part1_baseline_model.py
PART2 := part2_optimizations.py
PART3_A := part3_deployment.py
PART3_B := part3_modeling.py
PROFILER := performance_profiler.py
FRAMEWORK := optimization_framework.py

# Training
train:
	$(PYTHON) $(PART1)
	@echo "Baseline model training completed"

# Optimization
optimize:
	$(PYTHON) $(PART2)
	@echo "Hardware-aware optimizations completed"

# Benchmarking
benchmark:
	$(PYTHON) $(PROFILER)
	@echo "Performance benchmarking completed"

# Deployment (Track A)
deploy:
	@if [ -z "$(PLATFORM)" ]; then \
		echo "Error: PLATFORM must be specified for deployment"; \
		echo "Available platforms: cpu_x86, arm_cortex_a, arm_cortex_m, gpu_mobile"; \
		exit 1; \
	fi
	$(PYTHON) $(PART3_A) --platform $(PLATFORM)
	@echo "Deployment to $(PLATFORM) completed"

# Simulation (Track B)
simulate:
	@if [ -z "$(SIMULATOR)" ]; then \
		echo "Error: SIMULATOR must be specified for simulation"; \
		echo "Available simulators: qemu, renode, webgpu"; \
		exit 1; \
	fi
	$(PYTHON) $(PART3_B) --simulator $(SIMULATOR)
	@echo "Simulation with $(SIMULATOR) completed"

# Analysis
analyze:
	@echo "Generating optimization analysis report..."
	$(PYTHON) $(FRAMEWORK) --generate-report
	@echo ""
	@echo "Analysis report generated successfully!"
	@echo "View report with: cat results/optimization_analysis.txt"

# Testing
test:
	$(PYTHON) -m pytest tests/ -v
	@echo "All tests completed"

# Code quality
lint:
	flake8 *.py
	black --check *.py
	@echo "Code quality check completed"

format:
	black *.py
	@echo "Code formatted"

# Create directories
setup:
	mkdir -p optimized_models
	mkdir -p logs
	mkdir -p results
	mkdir -p platform_configs
	@echo "Directory structure created"

# Clean generated files
clean:
	rm -rf __pycache__/
	rm -rf *.pyc
	rm -rf .pytest_cache/
	rm -rf optimized_models/*.keras
	rm -rf optimized_models/*.tflite
	rm -rf logs/*.log
	rm -rf results/*.json
	rm -rf results/*.csv
	@echo "Generated files cleaned"

# Full pipeline (Track A)
pipeline-a: setup train optimize benchmark deploy analyze
	@echo "Track A pipeline completed successfully"

# Full pipeline (Track B)
pipeline-b: setup train optimize benchmark simulate analyze
	@echo "Track B pipeline completed successfully"

# Development targets
dev-setup: setup
	@echo "Development environment ready"

# Documentation generation
docs:
	@echo "Generating documentation..."
	$(PYTHON) -c "import part1_baseline_model; import part2_optimizations; import inspect; print('Documentation generated in docs/')"

# Performance validation
validate:
	$(PYTHON) $(PROFILER) --validate
	@echo "Performance validation completed"

# Platform-specific targets
deploy-cpu: deploy
	$(MAKE) deploy PLATFORM=cpu_x86

deploy-arm-a: deploy
	$(MAKE) deploy PLATFORM=arm_cortex_a

deploy-arm-m: deploy
	$(MAKE) deploy PLATFORM=arm_cortex_m

deploy-gpu: deploy
	$(MAKE) deploy PLATFORM=mobile_gpu

# Simulation-specific targets
simulate-qemu: simulate
	$(MAKE) simulate SIMULATOR=qemu

simuate-renode: simulate
	$(MAKE) simulate SIMULATOR=renode

simulate-webgpu: simulate
	$(MAKE) simulate SIMULATOR=webgpu