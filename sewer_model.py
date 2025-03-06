'''
Simulation code based on PySWMM engine
getting data for a GP prediction
Multi-input GPR for forecasting flow
GPflow library is used - the results are made for the UDM2025 conference 
Author: Mohsen Rezaee
Version: 6.1
Date: 18/12/2024
'''
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gpflow
from pyswmm import Simulation, Nodes, RainGages
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
from scipy.stats import norm

start_time = datetime.datetime(2022, 1, 1, 0, 0, 0)
end_time = datetime.datetime(2023, 1, 1, 0, 0, 0)

train_range = [(2, 400), (400, 4325)]
test_range = (4325, 8750)

#gpflow.config.set_default_float(tf.float32) # setting the default to float32

def collecting_data():
    with Simulation(r'model_2_2.inp') as sim:
        # defining start and end times instead of getting it from the simulator
        sim.start_time = start_time 
        sim.end_time = end_time
        
        print("Simulation info:")
        flow_units = sim.flow_units
        system_units = sim.system_units
        print(f'Flow Units: {flow_units} - System Units: {system_units}')
        print("Start Time: {}".format(sim.start_time))
        print("End Time: {}".format(sim.end_time))

        simtime = [] 
        hours = []
        rainfall = [] 
        WWTP = Nodes(sim)["WWTP"]
        WWTP_inflow = []
        CSO = Nodes(sim)["CSO_outfall"]
        CSO_discharge = []
        rain_gage = RainGages(sim)["RG1"] 

        sim.step_advance(180)    
        for ind, step in enumerate(sim):
            if ind % 20 == 0:    
                WWTP_inflow.append(WWTP.total_inflow)
                rainfall.append(rain_gage.rainfall)
                CSO_discharge.append(CSO.total_inflow)
                current_hours = (sim.current_time - start_time).total_seconds() / 3600
                hours.append(current_hours)
                simtime.append(sim.current_time)
    return hours, WWTP_inflow, CSO_discharge, rainfall

def Noise(WWTP_inflow):
    np.random.seed(100) # fixing the random values 
    
    noisy_data = WWTP_inflow.copy()
    noise_std = 0.8 # L/s
    noise_y_train = np.random.normal(0, noise_std, size=len(noisy_data))
    noise_y_train = np.clip(noise_y_train, -3 * noise_std, 3 * noise_std)
    noisy_data = noisy_data + noise_y_train
    
    return noisy_data

def preparing_data(hours, WWTP_inflow, CSO_discharge, rainfall):
    # removing the lag by two hours
    rainfall.insert(0, 0)
    rainfall.pop()
    
    # creating multi-dimensional input
    X_multi = np.column_stack([hours, rainfall])
    y = np.array(WWTP_inflow).reshape(-1, 1)
    
    train_indices = [] 
    for start, end in train_range:
        train_indices.extend(range(start, end))
    test_indices = range(*test_range)
    
    X_train = X_multi[train_indices]
    y_train = y[train_indices]
    X_test = X_multi[test_indices]
    y_test = y[test_indices]
    
    X_test_original = X_test.copy()
    X_train_original = X_train.copy()
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    
    return (X_train_scaled, X_test_scaled, y_train_scaled, 
            y_test_scaled, scaler_y, X_train_original, X_test_original)

def train_predict_gpflow(X_train, X_test, y_train):
    y_train = y_train.reshape(-1, 1)
    # GPflow kernel configuration
    kernel_hourly = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(lengthscales=0.005, variance=1), 
                                      period=1)
    kernel_daily = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(lengthscales=0.007, variance=1), 
                                      period=24)
    kernel_rain = gpflow.kernels.Matern12(active_dims=[1])
    kernel_noise = gpflow.kernels.White()
    
    kernel_hourly.active_dims = [0]
    kernel_daily.active_dims = [0]
    
    kernel = kernel_hourly + kernel_daily + kernel_rain + kernel_noise
    
    #gpflow.set_trainable(kernel_hourly.base_kernel.lengthscales, False)
    #gpflow.set_trainable(kernel_daily.base_kernel.lengthscales, False)
    gpflow.set_trainable(kernel_hourly.period, False)
    gpflow.set_trainable(kernel_daily.period, False)
    #gpflow.set_trainable(kernel_hourly.base_kernel.variance, False)
    #gpflow.set_trainable(kernel_daily.base_kernel.variance, False)

    # Convert to TensorFlow tensors
    X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float64)    # if the stability is a matter, we can use float64
    y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float64)
    X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float64)

    # Create GPflow model
    model = gpflow.models.GPR(data=(X_train_tf, y_train_tf), 
                               kernel=kernel)
    
    # Optimize hyperparameters
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, 
                 model.trainable_variables, 
                 method='L-BFGS-B')
    
    # printing the kernel parameters 
    for i, sub_kernel in enumerate(model.kernel.kernels):
        print(f"Sub-kernel {i}: {type(sub_kernel).__name__}")
        if isinstance(sub_kernel, gpflow.kernels.Periodic):
            lengthscale_p = sub_kernel.base_kernel.lengthscales.numpy()
            variance_p = sub_kernel.base_kernel.variance.numpy()
            period_p = sub_kernel.period.numpy()
            print(f"  Lengthscale of the Periodic kernel is {lengthscale_p}, the variance is {variance_p} and period is {period_p}")
        elif hasattr(sub_kernel, 'lengthscales'):
            print(f"  Lengthscales: {sub_kernel.lengthscales.numpy()}")
        if hasattr(sub_kernel, 'variance'):
            print(f"  Variance: {sub_kernel.variance.numpy()}")

    # Make predictions
    mean_test, var_test = model.predict_f(X_test_tf)
    mean_train, var_train = model.predict_f(X_train_tf)
    
    # Get std dev
    std_test = np.sqrt(var_test.numpy())
    std_train = np.sqrt(var_train.numpy())
    
    return mean_test.numpy(), std_test, mean_train.numpy(), std_train

def model_check(y_test, y_pred, std, scaler_y):
    # checking the fit of the model
    y_test_original = scaler_y.inverse_transform(y_test).ravel()
    y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    std_original = std * scaler_y.scale_
    
    lower_bound = y_pred_original - 1.96 * std_original.ravel()
    upper_bound = y_pred_original + 1.96 * std_original.ravel()
    
    points_inside = np.sum((y_test_original > lower_bound) &
                            (y_test_original < upper_bound))
    coverage = points_inside / len(y_test_original)
    
    #print(f'std dev original shape is {std_original.shape}')
    #print(f'the shape of y_test_original is {y_pred_original.shape}')
    MSE = np.mean((y_test_original.ravel() - y_pred_original.ravel())**2)
    RMSE = np.sqrt(MSE)
    
    print(f'The RMSE value is {RMSE: .2f} L/s')
    print(f'The coverage of the test data is {coverage*100: .1f}%')
    return RMSE, coverage 

def plot(X_test_scaled, y_test, y_pred, std, scaler_y, X_test_original,
         X_train_scaled, y_pred_train, std_train, X_train_original, y_train_scaled, 
         start_date):

    zoom_start = start_date + datetime.timedelta(hours=4600)
    zoom_end = start_date + datetime.timedelta(hours=5000)
    
    # transforming to the original scale 
    y_test_original = scaler_y.inverse_transform(y_test).ravel()
    y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    y_train_original = scaler_y.inverse_transform(y_train_scaled.reshape(-1 ,1)).ravel()
    y_pred_train_original = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()
    std_original = std * scaler_y.scale_
    std_train_original = std_train * scaler_y.scale_
    
    # defining time axis
    date_test = [start_date + datetime.timedelta(hours=h) for h in X_test_original[:, 0]]
    date_train = [start_date + datetime.timedelta(hours=h) for h in X_train_original[:, 0]]
    
    # zoom range 
    zoom_range_indices = [i for i, date in enumerate(date_test) if zoom_start <= date <= zoom_end]
    if not zoom_range_indices:
        raise  ValueError("NO DATA POINT IN THE ZOOM RANGE")
    zoom_range = (zoom_range_indices[0], zoom_range_indices[-1] + 1)
    
    # plotting a normal distribution for a point of interest 
    specific_date = zoom_start + datetime.timedelta(hours=198)    # defining the position of the point of interest 
    specific_index = min(range(len(date_test)), key=lambda i: abs(date_test[i] - specific_date))
    #matched_date = date_test[specific_index]
    specific_mean = y_pred_original[specific_index]
    specific_std = std_original[specific_index]
    actual_value_specific_point = y_test_original[specific_index]
    
    lower_bound_GP_dist = specific_mean - 1.96 * specific_std
    upper_bound_GP_dist = specific_mean + 1.96 * specific_std
    
    x = np.linspace(specific_mean - 4 * specific_std, specific_mean + 4 * specific_std, 1000).ravel()   # plotting +- 4 standard deviation 
    gaussian = norm.pdf(x, specific_mean, specific_std)
     
    plt.figure(figsize=(12, 6))
    plt.plot(x, gaussian, 'r-', lw=2)
    #plt.fill_between(x, gaussian, where=(x >= lower_bound_GP_dist) & (x <= upper_bound_GP_dist),
    #             color='none', edgecolor='black', hatch='//', label='95% Credible Interval')
    plt.title(f'Gaussian Distribution at {specific_date}', fontsize=22, fontweight='bold')
    plt.xlabel('WWTP Inflow (L/s)', fontsize=20)
    plt.ylabel('Probability Density', fontsize=20)
    plt.axvline(x=actual_value_specific_point, color='g', linestyle='--', label='Actual Value')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16) 
    plt.legend(fontsize=18)
    
    # main figure 
    fig, ax = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # main plot 
    ax[0].scatter(date_test, y_test_original, c='b', label='Actual Data Points', s=20)
    ax[0].plot(date_test, y_pred_original, 'r-', label= 'Prediction GPR Line')
    ax[0].fill_between(date_test, y_pred_original 
                     - 1.96 * std_original.ravel(), y_pred_original 
                     + 1.96 * std_original.ravel(), alpha=0.2, color='r')
    
    ax[0].scatter(date_train, y_train_original, c='b', s=20)
    ax[0].plot(date_train, y_pred_train_original, 'c-', label='Training GPR Line')
    ax[0].fill_between(date_train, y_pred_train_original 
                     - 1.96 * std_train_original.ravel(), y_pred_train_original 
                     + 1.96 * std_train_original.ravel(), alpha=0.2, color='c')
    ax[0].axvspan(date_test[zoom_range[0]], date_test[zoom_range[1]], color='yellow', alpha=0.3, label='Zoomed Region')
    #ax[0].set_xlabel('Date', fontsize=20)
    ax[0].set_ylabel('Flow (L/s)', fontsize=18)
    ax[0].set_title('Gaussian Processes Regression, WWTP Inflow Prediction', fontsize=22)
    ax[0].legend(fontsize=18)
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax[0].tick_params(axis='x', rotation=45)
    
    # zoomed in plot 
    zoom_dates = date_test[zoom_range[0]:zoom_range[1]]
    zoom_y_test = y_test_original[zoom_range[0]:zoom_range[1]]
    zoom_y_pred = y_pred_original[zoom_range[0]:zoom_range[1]]
    zoom_std = std_original[zoom_range[0]:zoom_range[1]].ravel()    
    
    ax[1].scatter(zoom_dates, zoom_y_test, c='b', label='Actual data points', s=40)
    ax[1].plot(zoom_dates, zoom_y_pred, 'r-', label='Prediction GPR line', linewidth=2)
    ax[1].fill_between(zoom_dates, zoom_y_pred - 1.96 * zoom_std, zoom_y_pred + 1.96 * zoom_std, 
                       alpha=0.3, color='r', label='95% Credible Interval')
    ax[1].set_xlabel('Date', fontsize=16)
    ax[1].set_ylabel('Flow (L/s)', fontsize=18)
    ax[1].set_title('Zoomed-in Section with Credible Intervals', fontsize=18)
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax[1].tick_params(axis='x', rotation=45)

    # Add vertical line for the specific point
    ax[1].axvline(x=specific_date, color='c', linestyle='--', label='Point of Interest')
    ax[1].legend(fontsize=14)
    
    plt.tight_layout()
    
    plt.show()

def main():
    start_t = time.time()
    
    # Data collection and preparation
    hours, WWTP_inflow, CSO_discharge, rainfall = collecting_data()
    WWTP_inflow = Noise(WWTP_inflow)
    
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_y, \
    X_train_original, X_test_original = preparing_data(hours, WWTP_inflow, CSO_discharge, rainfall)
    
    # Prediction using GPflow
    y_predict_scaled, std_test, y_predict_train_scaled, std_train = train_predict_gpflow(X_train_scaled, X_test_scaled, y_train_scaled)
    
    # Model checking
    RMSE, coverage = model_check(y_test_scaled, y_predict_scaled, std_test, scaler_y)
    
    end_t = time.time()
    print(f'Total running time was {round(end_t - start_t, 1)} seconds')
    
    # Plotting
    plot(X_test_scaled, y_test_scaled, y_predict_scaled, std_test, scaler_y, X_test_original,
         X_train_scaled, y_predict_train_scaled, std_train, X_train_original, y_train_scaled, 
         start_date=start_time)

if __name__ == "__main__":
    main()
