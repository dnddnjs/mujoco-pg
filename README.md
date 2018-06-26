# pytorch-trpo
PyTorch implementation of Trust Region Policy Optimization

# Train
* **algorithm**: PG, NPG, TRPO
* **env**: Ant-v2, HalfCheetah-v2, Hopper-v2, Humanoid-v2, HumanoidStandup-v2, InvertedPendulum-v2, Reacher-v2, Swimmer-v2, Walker2d-v2
~~~
python train.py --algorithm "algorithm name" --env "environment name"
~~~