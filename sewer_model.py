'''
Simulation code based on PySWMM engine
getting data for a GP prediction
setting up the kernel 
Author: Mohsen Rezaee
Version: 3
Date: 24/04/2024
'''
import time
import datetime
start_t = time.time()
import pickle
from pyswmm import Simulation, Nodes, Links
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic, Matern, WhiteKernel, DotProduct
import numpy as np
import matplotlib.pyplot as plt 

start_t_p1 = time.time()
"""
#first part
with Simulation(r'model_2_1.inp') as sim:
    # defining start and end times instead of getting it from the simulator
    sim.start_time = datetime.datetime(2023, 1, 1, 0, 0, 0) # (year, month, day, hour, minute, second)
    sim.end_time = datetime.datetime(2023, 7, 1, 0, 0, 0)
    
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
    
    # Launch a simulation!
    sim.step_advance(180)    # step_advance * steps (here 100) = time intervals of the results 
    for ind, step in enumerate(sim):
        if ind % 20 == 0:    # number of time steps for giving results in each running period, here it is 20 
          WWTP_inflow.append(WWTP.total_inflow)
          #ST_flooding.append(storage_tank.flooding)
          CSO_discharge.append(CSO.total_inflow)
          
          simtime.append(sim.current_time)
# saving the produced data 
data_from_pyswmm = {'WWTP_inflow': WWTP_inflow, 
                    'CSO_discharge': CSO_discharge,
                    'simtime': simtime}
with open('sim_data_2013to2025_1hr_intervals.pkl', 'wb') as f:
  pickle.dump(data_from_pyswmm, f)
"""
#Loading data
with open('sim_data_2013to2025_1hr_intervals.pkl', 'rb') as f: 
  loaded_data = pickle.load(f)
WWTP_inflow = loaded_data['WWTP_inflow']
CSO_discharge = loaded_data['CSO_discharge']
end_t_p1 = time.time()
print(f'The loading time was {round(end_t_p1 - start_t_p1)} seconds')
# --------------------------------------------------
start_t_p2 = time.time()
strt1, strt2 = 1, 2001
end1, end2 = 2000, 3000
GP_dict1 = {t: (influent, overflows) for t, influent, overflows in zip(range(strt1, end1), WWTP_inflow[strt1:end1], CSO_discharge[strt1:end1])}  
GP_dict2 = {t: (influent, overflows) for t, influent, overflows in zip(range(strt2, end2), WWTP_inflow[strt2:end2], CSO_discharge[strt2:end2])}  
GP_dict = {**GP_dict1, **GP_dict2}

GP_dict_test = {t: (influent, overflows) for t, influent, overflows in zip((range(3001, 4000)), WWTP_inflow[3001:4000], CSO_discharge[3001:4000])}

hours = np.array([hr for hr, _ in GP_dict.items()]).reshape(-1, 1)
WWTP_influent = np.array([inf for inf, _ in GP_dict.values()]).reshape(-1, 1)

# Split the data into train and test sets0
X_train, y_train = (hours, WWTP_influent)

#I add some test points, myself 
X_test = np.array([thr for thr, _ in GP_dict_test.items()]).reshape(-1, 1)
y_test = np.array([tinf for tinf, _ in GP_dict_test.values()]).reshape(-1, 1)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the kernel
daily_kernel = ExpSineSquared (length_scale=0.01, periodicity=3, length_scale_bounds=(5e-3, 1), periodicity_bounds="fixed")
Longterm_variations = ExpSineSquared (length_scale=0.002, periodicity=72, length_scale_bounds=(1e-3, 1e-1), periodicity_bounds="fixed")

kernel = daily_kernel + 10 * Longterm_variations
# Instantiate the Gaussian Process Regression model
GPR = GaussianProcessRegressor(kernel=kernel, alpha=0.1 , normalize_y=True)
GPR.fit(X_train_scaled, y_train)

kernel_params = GPR.kernel_.get_params()  # getting kernel parameters 
print(kernel_params)

WWTP_influent_predict, std = GPR.predict(X_test_scaled, return_std=True)

# Sort the data for plotting
sorted_indices = np.argsort(X_test.ravel())
X_test_sorted = X_test[sorted_indices]
y_test_sorted = y_test[sorted_indices]
WWTP_inflow_predict_sorted = WWTP_influent_predict[sorted_indices] 

end_t_p2 = time.time()
print(f'The running time for our GPR was {round(end_t_p2 - start_t_p2)} seconds')

# Plot the predictions
plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, c='y', label='Train Data', s=50)
plt.scatter(X_test_sorted, y_test_sorted, c='b', label='Actual', s=50)
plt.scatter(X_test_sorted, WWTP_inflow_predict_sorted, c='r', label='Predicted', s=50)

# plotting the line 
X_plot = np.linspace(np.min(np.concatenate((X_train, X_test))),
                     np.max(np.concatenate((X_train, X_test))), 1000).reshape(-1, 1)
X_plot_scaled = scaler.transform(X_plot)
y_plot, std_plot = GPR.predict(X_plot_scaled, return_std=True)
plt.plot(X_plot, y_plot, 'k--', label='GPR Line (test data)')
plt.fill_between(X_plot.ravel(), y_plot.ravel() - 1.96 * std_plot.ravel(), y_plot.ravel() + 1.96 * std_plot.ravel(), alpha=0.2, color='k')

# final plotting 
plt.xlabel('Hours')
plt.ylabel('WWTP Inflow')
plt.title('Gaussian Process Regression: WWTP Inflow Prediction')
plt.legend()

end_t = time.time()
print(f'Total running time was {round(end_t - start_t, 2)} seconds')

plt.show()