# Able to run the script using these steps
1) Using Anaconda with Isaac Lab: follow steps with this to install it via pip.
https://isaac-sim.github.io/IsaacLab/source/setup/installation/pip_installation.html

2) Activate this environment
```
conda activate isaaclab
```

3) install this extension within it
```
pip install -e .
cd exts/Ext_Robot_Arm
python -m pip install -e .
```

4) Run the test script
```
python scripts/rsl_rl/train.py --task=Template-Isaac-Velocity-Rough-Anymal-D-v0
```