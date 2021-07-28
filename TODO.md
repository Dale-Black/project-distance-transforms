# Roadmap
- [ ] Time CPU loss functions
  * Julia
	* `dice_loss`
    * `hd_loss`
  * Python
	* `dice_loss`
	* `hd_loss`
- [ ] Time GPU loss functions
  * Julia
	* `dice_loss`
	* `hd_loss`
  * Python
    * `dice_loss`
	* `hd_loss`
- [ ] Time CPU DTs
  * Julia
	* ~~`squared_euclidean_distance_transform` CPU threaded~~
	* ~~`squared_euclidean_distance_transform` CPU~~
	* ~~`chamfer_distance_transform`~~
	* ~~`euclidean_distance_transform`~~
  * Python
	* `euclidean_distance_transform`
- [ ] Time GPU DT
	-  `squared_euclidean_distance_transform` GPU
- [ ] Determine dataset(s) to use for training
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
