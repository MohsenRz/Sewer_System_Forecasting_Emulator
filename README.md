# Sewer_System_Forecasting_Emulator

This project forecasts flow, depth, and detects anomalies in sewer systems using Gaussian Process Regression (GPR). 
We implement predictive models based on data from flowmeters and simulation outputs using the GPflow library.

Data Generation: PySWMM engine (Python interface for EPA SWMM) simulates flow in a hypothetical sewer system with one Wastewater Treatment Plant (WWTP) and Combined Sewer Overflow (CSO) point

Modeling Approach: Multi-input Gaussian Process Regression with customized kernel configurations

Kernel Selection: We use a composite kernel approach to capture different flow characteristics:
- Periodic kernels for hourly patterns (kernel_hourly)
- Periodic kernels for daily patterns (kernel_daily)
- Matérn kernels for rainfall response (kernel_rain)
- White noise kernel for measurement uncertainty

Implementation
Built with Python using:
- GPflow (TensorFlow-based GP library)
- PySWMM for simulation
- TensorFlow for computation
- Matplotlib for visualization
