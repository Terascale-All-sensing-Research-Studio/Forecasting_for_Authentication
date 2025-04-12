## [AIxVR2024] Using Motion Forecasting for Behavior-Based Virtual Reality (VR) Authentication

## :boom::boom: *Update [Jan 2024]:* We won the  $\Large{Best Paper Award}$ at the 6th IEEE International Conference on Artificial Intelligence & extended and Virtual Reality ([AIxVR 2024](https://aivr.science.uu.nl/))!

<div align="center">
<img src="https://github.com/Terascale-All-sensing-Research-Studio/Forecasting_for_Authentication/blob/main/figs/teaser.jpg" width=100% height=100%>
</div>

Task-based behavioral biometric authentication of users interacting in virtual reality (VR) environments enables seamless continuous authentication by using only the motion trajectories of the person’s body as a unique signature. Deep learning-based approaches for behavioral biometrics show high accuracy when using complete or near complete portions of the user trajectory, but show lower performance when using smaller segments from the start of the task. Thus, any systems designed with existing techniques are vulnerable while waiting for future segments of motion trajectories to become available. In this work, we present the first approach that predicts future user behavior using Transformer-based forecasting and using the forecasted trajectory to perform user authentication. Our work leverages the notion that given the current trajectory of a user in a task based environment, we can predict the future trajectory of the user as they are unlikely to dramatically shift their behavior since it would preclude the user from successfully completing their task goal. Using the [publicly available 41-subject ball throwing dataset](https://github.com/Terascale-All-sensing-Research-Studio/VR-Biometric-Authentication/tree/main/Authentication/DatasetsNumpy). we show improvement in user authentication when using forecasted data. When compared to no forecasting, our approach reduces the authentication equal error rate (EER) by an average of __23.85%__ and a maximum reduction of __36.14%__.

If you find our work helpful please cite us:
```
@inproceedings{li2024using,
  title={Using motion forecasting for behavior-based virtual reality (vr) authentication},
  author={Li, Mingjun and Banerjee, Natasha Kholgade and Banerjee, Sean},
  booktitle={2024 IEEE International Conference on Artificial Intelligence and eXtended and Virtual Reality (AIxVR)},
  pages={31--40},
  year={2024},
  organization={IEEE}
}
```

## What the [Data](https://github.com/Terascale-All-sensing-Research-Studio/Forecasting_for_Authentication/tree/main/ball_throwing_data) looks like.
<div align="left">
<img src="https://github.com/Terascale-All-sensing-Research-Studio/Forecasting_for_Authentication/blob/main/figs/user_full_trajectory.gif" width=60% height=60%>
</div>

## Installation

Code tested using Ubutnu __20.04__ and python __3.8__.

We recommend using virtualenv. The following snippet will create a new virtual environment, activate it, and install deps.
```bash
sudo apt-get install virtualenv && \
virtualenv -p python venv && \
source venv/bin/activate && \
git clone https://github.com/Atlas-Li/AIxVR2024_Forecasting_for_Authentication.git && \
pip install -r requirements.txt
```

## Training

### Authentication without forecasting

Navigate into the `python/authentication` directory.
```
cd python/authentication
```

Using FCN

```
python FCN_train.py -u <user_id> --ws <window_size>
```

Using Transformer encoder

```
python TFencoder_train.py -u <user_id> --ws <window_size>
```

Use the flag ```-l``` if save the log file.

### Authentication with forecasting

Navigate into the `python/forecast_authentication` directory.
```
cd python/forecast_authentication
```

Training


```
python main.py -u <user_id> --seq_len 30 --label_len 10 --pred_len 20 --classification_model FCN
```

You may want to change the number of timestamps of the input (```--sep_len```), the overlapped number of timestamps (```--label_len```), and the number of forecasting trajectory (```--pred_len```).

Change the authentication model

1. ```--classification_model FCN``` if using FCN;
2. ```--classification_model TF```  if using Transformer encoder.

Use the flag ```-l``` if save the log file.

## Docker Image

https://hub.docker.com/r/atlasli/aixvr2024_forecast_for_vr_authentication
