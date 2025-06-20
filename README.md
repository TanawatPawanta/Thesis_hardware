For joints trajectories: src/cascade_controller/include (use v4_2)

The trajectories are seperate into 2 plase

1.Init (Set initial joint position in simulation to first index of below trajectories)

  q_init_data_step10cmV4_2.npz -> joint position and velocity trajectory
  
2.Walk

  q_data_step10cmV4_2.npz -> joint position trajectory
  
  qd_data_step10cmV4_2.npz -> joint velocity trajectory

How to load datas: u can find load function(LoadDatas) at **src/cascade_controller/scripts/MotorControlGroupPublisher.py**
