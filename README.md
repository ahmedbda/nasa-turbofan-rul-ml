# NASA Turbofan Engine RUL prediction model
A Python machine learning project with the NASA C-MAPSS dataset to work on my data engineering skills

## Project goal

This project uses machine learning (Random Forest) to predict the RUL (remaining useful life) of turbojet engines based on sensor data provided by the NASA C-MAPSS dataset.  

Given the history of sensor readings (temperature, pressure, fan speed, and more), the goal is to predict exactly how many flight cycles an engine has left before it breaks down

## Results

### Performance

The model was evaluated using the RMSE (root mean squared error) metric on a separated test set (with unseen engines)

| Dataset | RMSE score | Interpretation |
| Validation set 20% | 44.76 cycles | Average performance |
| Test set | 35.77 cycles | Good performance |

Conclusion: The model performs exceptionally well when the engine approaches failure which is the most critical phase for maintenance decisions

### Visualization

I hypothesized about the properties of a turbofan engine to select the most relevant sensors
* Temperatures (sensors 2 and 4): rise due to efficiency loss 
* Pressure and fan speed (sensors 7 and 12): drop due to wear and tear

<p align="center"><img src="img/first_turbofan_behavior.png"></p>

I then applied a rolling mean with a window of 5 cycles to extract the true degradation trend and to calm the noise

<p align="center"><img src="img/first_turbofan_smoothed.png"></p>

I chose a Random Forest regression for its ability to handle non-linear calculations and its robustness to noise  
Mathematically, the prediction $\hat{y}$ for a given input $x$ is the average of the predictions of $K$ individual decision trees $T_k$:

$$\hat{y} = \frac{1}{K} \sum_{k=1}^{K} T_k(x)$$

Here I am using 100 decision trees and a fixed random generator for reproducibility.

<p align="center"><img src="img/model_ontest.png"></p>

And now the final predictions of the model on the real test set with the settings mentionned

<p align="center"><img src="img/model_ontrain.png"></p>


## How to run the calculated model

The trained model that I have calculated with this project is included in this repository as a compressed file for convenience purposes  

To use it in your own Python scripts, simply locate the file named model.zip in the root directory and extract it, you should then see a file named rf_model_rul.pkl

## Credits & references

Dataset: [NASA Turbofan Jet Engine Data Set](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) by behrad3d

Paper: A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM, 2008