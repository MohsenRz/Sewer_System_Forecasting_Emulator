'''
Simulation code based on PySWMM engine
getting data for a GP prediction
noise adjustment
Author: Mohsen Rezaee
Version: 4    
Date: 08/10/2024
'''
import time
start_t = time.time()
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic, Matern, WhiteKernel
import sklearn.metrics
import numpy as np
import numpy.random as nprand
import matplotlib.pyplot as plt 
from scipy.stats import norm

start_t_p1 = time.time()
#first part
import datetime
from pyswmm import Simulation, Nodes, Links, RainGages

start_time = datetime.datetime(2022, 1, 1, 0, 0, 0) # (year, month, day, hour, minute, second)
end_time = datetime.datetime(2023, 1, 1, 0, 0, 0)

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
    
    # getting the WWTP's inflow data 
    WWTP = Nodes(sim)["WWTP"]
    WWTP_inflow = []
    # getting the data of the Tank
    storage_tank = Nodes(sim)["St1"]
    ST_flooding = []
    # getting the CSO data 
    CSO = Nodes(sim)["CSO_outfall"]
    CSO_discharge = []
    #rain gauge 
    rain_gage = RainGages(sim)["RG1"] 

    # Launch a simulation!
    sim.step_advance(180)    # step_advance * steps (here 100) = time intervals of the results 
    for ind, step in enumerate(sim):
        if ind % 20 == 0:    # number of time steps for giving results in each running period, here it is 20 
          WWTP_inflow.append(WWTP.total_inflow)
          #ST_flooding.append(storage_tank.flooding)
          CSO_discharge.append(CSO.total_inflow)
          
          simtime.append(sim.current_time)
end_t_p1 = time.time()
print(f"The simulation time was {round(end_t_p1 - start_t_p1)} seconds")

start_t_p2 = time.time()
strt1, strt2 = 2, 21
end1, end2 = 20, 4324  # 6 mo
GP_dict1 = {t: (influent, overflows) for t, influent, overflows in zip(range(strt1, end1), WWTP_inflow[strt1:end1], CSO_discharge[strt1:end1])}  
GP_dict2 = {t: (influent, overflows) for t, influent, overflows in zip(range(strt2, end2), WWTP_inflow[strt2:end2], CSO_discharge[strt2:end2])}  
GP_dict = {**GP_dict1, **GP_dict2}

GP_dict_test = {t: (influent, overflows) for t, influent, overflows in zip((range(4325,8750)), WWTP_inflow[4325:8750], CSO_discharge[4325:8750])} # 6 mo - 6 mo 

hours = np.array([hr for hr, _ in GP_dict.items()]).reshape(-1, 1)
WWTP_influent = np.array([inf for inf, _ in GP_dict.values()]).reshape(-1, 1)

# adding noise to the input train data 
noise_std = 0.8 # this is the standard deviation of the normal distribution I've got
noise_WWTP_train = nprand.normal(0, noise_std, size=WWTP_influent.shape)
noise_WWTP_train = np.clip(noise_WWTP_train, -3 * noise_std, 3 * noise_std)
WWTP_influent_noisy = WWTP_influent + noise_WWTP_train

# Split the data into train and test sets
X_train, y_train = (hours, WWTP_influent_noisy)

#I add some test points, myself 
X_test = np.array([thr for thr, _ in GP_dict_test.items()]).reshape(-1, 1)
y_test = np.array([tinf  for tinf, _ in GP_dict_test.values()]).reshape(-1, 1)

# adding noise to the input test data 
noise_WWTP_test = nprand.normal(0, noise_std, size=y_test.shape)
noise_WWTP_test = np.clip(noise_WWTP_test, -3 * noise_std, 3 * noise_std) 
y_test_noisy = y_test + noise_WWTP_test 
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the kernel
shortterm_variations = ExpSineSquared (length_scale=0.006, periodicity=3, length_scale_bounds="fixed", periodicity_bounds="fixed") # hourly changes 
midterm_variations = ExpSineSquared (length_scale=0.005, periodicity=24, length_scale_bounds="fixed", periodicity_bounds="fixed") # daily changes 
Longterm_variations = ExpSineSquared (length_scale=0.1, periodicity=24*30, length_scale_bounds=(0.01, 1), periodicity_bounds="fixed") # monthly chnages 
new_kernel = RBF(length_scale=0.1, length_scale_bounds=(0.01,1)) # new for precipitation
noise_k = WhiteKernel(noise_level=0.01, noise_level_bounds=(0.004,0.5))
mat_kernel = Matern(0.001, (0.000001,1000), 2.5)

seven_day_k = ExpSineSquared(length_scale=0.01, periodicity=3, length_scale_bounds=(0.01, 0.1), periodicity_bounds="fixed")
kernel_24months_1hrs = shortterm_variations + midterm_variations + Longterm_variations + noise_k + new_kernel
# Instantiate the Gaussian Process Regression model
GPR = GaussianProcessRegressor(kernel=kernel_24months_1hrs, alpha=1e-8, normalize_y=True) 
GPR.fit(X_train_scaled, y_train)

kernel_params = GPR.kernel_.get_params()  # getting kernel parameters 
print(kernel_params)

WWTP_influent_predict, std = GPR.predict(X_test_scaled, return_std=True)

# Sort the data for plotting
sorted_indices = np.argsort(X_test.ravel())
X_test_sorted = X_test[sorted_indices]
y_test_sorted = y_test_noisy[sorted_indices]
WWTP_inflow_predict_sorted = WWTP_influent_predict[sorted_indices] 

end_t_p2 = time.time()
print(f'The running time for our GPR was {round(end_t_p2 - start_t_p2)} seconds')

# posterior prediction checks 
# Coverage statistics 
lower_bound = (WWTP_influent_predict - 1.96 * std).ravel()
upper_bound = (WWTP_influent_predict + 1.96 * std).ravel()
points_inside = np.sum((y_test_noisy.ravel() > lower_bound) & (y_test_noisy.ravel() < upper_bound))
coverage_value = points_inside / len(y_test_noisy)
print(f'The covarage percetage of the test data was {100*coverage_value}%')
# Normalized Root Mean Square Error
MSE = sklearn.metrics.mean_squared_error(y_test_noisy.ravel(), WWTP_influent_predict.ravel())
RMSE = np.sqrt(MSE)
range = np.max(y_test_noisy) - np.min(y_test_noisy)
NRMSE = RMSE / range 
print(f'The RMSE value is {RMSE}')
print(f'The NRMSE value is {NRMSE}')

# converting date and time for plotting 
X_train_dates = [start_time + datetime.timedelta(hours=int(hr)) for hr in X_train.ravel()]
X_test_dates = [start_time + datetime.timedelta(hours=int(hr)) for hr in X_test_sorted.ravel()]

# Generate the plot line data
X_plot = np.linspace(np.min(np.concatenate((X_train, X_test))), np.max(np.concatenate((X_train, X_test))), 1000).reshape(-1, 1)
X_plot_scaled = scaler.transform(X_plot)
y_plot, std_plot = GPR.predict(X_plot_scaled, return_std=True)

# Convert X_plot to dates for plotting
X_plot_dates = [start_time + datetime.timedelta(hours=int(hr)) for hr in X_plot.ravel()]

# Plot the predictions
plt.figure(figsize=(12, 6))
plt.scatter(X_train_dates, y_train, c='y', label='Train Data', s=20)
plt.scatter(X_test_dates, y_test_sorted, c='b', label='Actual', s=20)
#plt.scatter(X_test_dates, WWTP_inflow_predict_sorted, c='r', label='Predicted', s=20)
plt.plot(X_plot_dates, y_plot, 'k--', label='GPR Line')
plt.fill_between(X_plot_dates, y_plot.ravel() - 1.96 * std_plot.ravel(), y_plot.ravel() + 1.96 * std_plot.ravel(), alpha=0.2, color='k')

# Add vertical line for the specific point
#plt.axvline(x=specific_date, color='r', linestyle='--', label='Point of Interest')

# Set the size of the values on the two axes
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# final plotting 
plt.xlabel('Date', fontsize=20)
plt.ylabel('WWTP Inflow (L/s)', fontsize=20)
plt.title('Gaussian Process Regression: WWTP Inflow Prediction', fontsize=22, fontweight= 'bold')

plt.legend(fontsize=18)

end_t = time.time()
print(f'Total running time was {round(end_t - start_t, 2)} seconds')

plt.show()
