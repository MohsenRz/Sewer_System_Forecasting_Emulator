# Sewer_System_Forecasting_Emulator
Using Gaussian processes for predicting flows and CSOs in a combines sewer system 

In this project we will get predictions based on the data given from flowmeters/simulator's output using a GP approach. 

At the first step, pyswwm engine is used as a simulator to generate data for a hypothetized skeleton sewer system, with one WWTP and CSO. 

We are using different kernels to catch different characteristics of the flow on short and long-term. 