# Deep Reinforcement Learning 

## Important info
- Dependencies are in `requirements.txt`
- Best performances (episodes recordings) are in `episodes/`
- Pretrained model weights are in `pretrained_models/`

## How to use it ?

1. Create new virtual environment  
```
python3 -m venv ./ venv
```
2. Install all dependencies 
```
pip3 install -r requirements.txt
``` 
3. Activate the newly created env
```
source venv/bin/activate
```
4. Test demo (with saved pretrained model weights and does not trigger episode recordings)
```
python3 demo.py
```
5. (Extra) Choose which agent you want to test (Notice: these commands will trigger the recording of episodes and will save the newly trained model weights)

    1. Random Agent (episode recordings are in `episodes/random_agent`)
        ```
        python3 random_agent.py
        ```
    2. Deep Q-Network CartPole agent (episode recordings are in `episodes/dqn_agent`)
        ```
        python3 dqn_agent.py
        ```
    3. Deep Q-Convolutional Network CartPole agent (episode recordings are in `episodes/conv_dqn_agent`)
        ```
        python3 conv_dqn_agent.py
        ```
    4. Deep Q-Convolutional Network MineRL agent 
        ```
        python3 minerl_agent.py
        ```

Finally, don't forget to deactivate the virtual environment after testing 
```
deactivate
```

## Developed by 
- Khaled ABDRABO p1713323 
- Jean BRIGNONE p1709655
