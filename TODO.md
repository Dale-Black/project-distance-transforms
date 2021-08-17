# Roadmap
- [x] Time CPU loss functions
  * Julia (Dale) [notebook](https://github.com/Dale-Black/project-distance-transforms/blob/master/julia/timing/loss_functions.jl)
	* ~~`dice_loss`~~
	* ~~`hd_loss`~~
  * Python (Ashwin) [notebook](https://github.com/Dale-Black/project-distance-transforms/blob/master/python/timing/python_loss_functions.ipynb)
	* ~~`dice_loss`~~
	* ~~`hd_loss`~~
- [x] Time GPU loss functions
  * Julia (Dale) [notebook](https://github.com/Dale-Black/project-distance-transforms/blob/master/julia/timing/loss_functions.jl)
	* ~~`dice_loss`~~
	* ~~`hd_loss`~~
  * Python (Ashwin)[notebook](https://github.com/Dale-Black/project-distance-transforms/blob/master/python/timing/python_loss_functions.ipynb)
	* ~~`dice_loss`~~
	* ~~`hd_loss`~~
- [ ] Time CPU DTs
  * Julia (Dale) [notebook](https://github.com/Dale-Black/project-distance-transforms/blob/master/julia/timing/dt.jl)
	* ~~`squared_euclidean` CPU threaded~~
	* ~~`squared_euclidean` CPU~~
	* ~~`chamfer`~~
	* ~~`euclidean`~~
  * Python (Ashwin)
	* `euclidean_distance_transform`
- [x] Time GPU DT (Dale) [notebook](https://github.com/Dale-Black/project-distance-transforms/blob/master/julia/timing/dt.jl)
	-  ~~`squared_euclidean` GPU~~
- [ ] Time step for various DTs [use this](https://github.com/ChrisRackauckas/PINN_Quadrature/blob/main/Examples/level_set_num.jl)
	* `squared_euclidean` GPU
	* `squared_euclidean` CPU threaded
	* `squared_euclidean` CPU
	* `chamfer`
	* `euclidean`
	* pure `dice_loss`
- [ ] Time epoch for various DTs
	* `squared_euclidean` GPU
	* `squared_euclidean` CPU threaded
	* `squared_euclidean` CPU
	* `chamfer`
	* `euclidean`
	* pure `dice_loss`
- [ ] Determine dataset(s) to use for training (Dale & Ashwin)
	* Use datasets from previous papers
- [ ] Compare convergence rate
	* `squared_euclidean` GPU
	* `euclidean`
	* pure `dice_loss`
- [ ] Compare metrics
	* `squared_euclidean` GPU
	* `euclidean`
	* pure `dice_loss`
- [ ] Write paper
  * ...
