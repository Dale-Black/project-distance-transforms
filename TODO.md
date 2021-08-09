# Roadmap
- [ ] Time CPU loss functions (Ashwin)
  * Julia
	* `dice_loss`
    * `hd_loss`
  * Python
	* `dice_loss`
	* `hd_loss`
- [ ] Time GPU loss functions (Dale)
  * Julia
	* `dice_loss`
	* `hd_loss`
  * Python
    * `dice_loss`
	* `hd_loss`
- [ ] Time CPU DTs (Dale)
  * Julia [notebook](https://github.com/Dale-Black/project-distance-transforms/blob/master/julia/timing/pluto_notebooks/cpu_dt.jl)
	* ~~`squared_euclidean_distance_transform` CPU threaded~~
	* ~~`squared_euclidean_distance_transform` CPU~~
	* ~~`chamfer_distance_transform`~~
	* ~~`euclidean_distance_transform`~~
  * Python (Ashwin)
	* `euclidean_distance_transform`
- [ ] Time GPU DT (Dale)
	-  `squared_euclidean_distance_transform` GPU
- [ ] Determine dataset(s) to use for training (Dale)
	* Use datasets from previous papers
- [ ] Time step for various DTs
	* `squared_euclidean_distance_transform` GPU
	* `squared_euclidean_distance_transform` CPU threaded
	* `squared_euclidean_distance_transform` CPU
	* `chamfer_distance_transform`
	* `euclidean_distance_transform`
	* pure `dice_loss`
- [ ] Time epoch for various DTs
	* `squared_euclidean_distance_transform` GPU
	* `squared_euclidean_distance_transform` CPU threaded
	* `squared_euclidean_distance_transform` CPU
	* `chamfer_distance_transform`
	* `euclidean_distance_transform`
	* pure `dice_loss`
- [ ] Compare convergence rate
	* `squared_euclidean_distance_transform` GPU
	* `euclidean_distance_transform`
	* pure `dice_loss`
- [ ] Compare metrics
	* `squared_euclidean_distance_transform` GPU
	* `euclidean_distance_transform`
	* pure `dice_loss`
- [ ] Write paper
  * ...
