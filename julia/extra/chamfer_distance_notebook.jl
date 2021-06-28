### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 3bec943c-d6d4-11eb-114a-81c372ddbd7c
begin
	let
		using Pkg
		Pkg.activate(mktempdir())
		Pkg.Registry.update()
		Pkg.add.([
			"ImageMorphology",
			"BenchmarkTools",
			"DistanceTransforms"
			])
	end
	
	using ImageMorphology
	using BenchmarkTools
	using DistanceTransforms
	using PlutoUI
end

# ╔═╡ e4105107-121a-4fd4-9090-764dc4a38129
md"""
# Simple Distance Transform
"""

# ╔═╡ c6e99dfb-00df-4217-aaf6-d34d9dd88391
TableOfContents()

# ╔═╡ 2bb02612-8e85-49f0-a680-7a4acbc24926
md"""
## Load test data
"""

# ╔═╡ 733bf0aa-aab9-4a76-bca3-b8fb5a15f0ca
begin
	a = rand([0,1], (4,4))
	b = rand([0,1], (2,2))
	x = rand([0,1], (112, 112, 96, 2, 1))
	y = rand([0,1], (112, 112, 96, 2, 1))
end;

# ╔═╡ 0fe2e0b6-2812-4daa-a3d1-dd6be6b0176f
a

# ╔═╡ 63ae0c38-d7bb-454d-9771-d788d9fa91f9
b

# ╔═╡ 147b796c-1cf3-4b4b-b819-a348bdd01ef6
a_bool = Bool.(a)

# ╔═╡ c6aa66e0-877d-4e85-ae24-3febaffef557
b_bool = Bool.(b)

# ╔═╡ c06fc005-f48e-4f9f-8ce8-a88e9c458fe5
begin 
	x_bool = Bool.(x)
	y_bool = Bool.(y)
end;

# ╔═╡ d4b055c3-c4cb-4a05-9529-a30858916f7f
md"""
## Examine `feature_transform`
"""

# ╔═╡ 2e3093d8-8a00-49dd-934f-3bd8e6534dab
ImageMorphology.feature_transform(a_bool)

# ╔═╡ 3d657311-428a-456d-a457-4bf314e2c7d7
ImageMorphology.feature_transform(b_bool)

# ╔═╡ 94ffcb2d-7cd4-4b1c-a07a-e2bde42bdd43
md"""
## Time `feature_transform`
"""

# ╔═╡ c92f21b2-5d84-495a-8981-af1a1fbd774e
@benchmark feature_transform(a_bool)

# ╔═╡ bba80cc8-f9e3-4523-9b7c-25489aa90d41
@benchmark feature_transform(x_bool)

# ╔═╡ a6107a3f-eecd-4b8c-a298-c9bdf3f5bbec
@benchmark feature_transform(y_bool)

# ╔═╡ 50a6a2d0-5436-446c-9b01-6624a60300ba
md"""
## Time `compute_dtm`
"""

# ╔═╡ e86f986d-b0cb-44b4-a0a3-17651037e01c
@benchmark DistanceTransforms.compute_dtm($x)

# ╔═╡ 01942fc2-e32b-437a-a3a3-7d2385d5f32a
md"""
## Try `chamfer_distance`
"""

# ╔═╡ 7d5f4bc1-455a-4ebb-b384-cc73c6e796c7
function chamfer_distance(img)
	w, h = size(img)
	dt = zeros(Int32, (w,h))
	# Forward pass
	x = 1
	y = 1
	if img[x,y] == 0
		dt[x,y] = 65535 # some large value
	end
	for x in 1:w-1
		if img[x+1,y] == 0
			dt[x+1,y] = 3 + dt[x,y]
		end
	end
	for y in 1:h-1
		x = 1
		if img[x,y+1] == 0
			dt[x,y+1] = min(3 + dt[x,y], 4 + dt[x+1,y])
		end
		for x in 1:w-2
			if img[x+1,y+1] == 0
				dt[x+1,y+1] = min(4 + dt[x,y], 3 + dt[x+1,y], 4 + dt[x+2,y], 3 + dt[x,y+1])
			end
		end
		x = w
		
		if img[x,y+1] == 0
			dt[x,y+1] = min(4 + dt[x-1,y], 3 + dt[x,y], 3 + dt[x-1,y+1])
		end
	end
	
	# Backward pass
	for x in w-1:-1:1
		y = h
		if img[x,y] == 0
			dt[x,y] = min(dt[x,y], 3 + dt[x+1,y])
		end
	end
	for y in h-1:-1:1
		x = w
		if img[x,y] == 0
			dt[x,y] = min(dt[x,y], 3 + dt[x,y+1], 4 + dt[x-1,y+1])
		end
		for x in 1:w-2
			if img[x+1,y] == 0
				dt[x+1,y] = min(dt[x+1,y], 4 + dt[x+2,y+1], 3 + dt[x+1,y+1], 4 + dt[x,y+1], 3 + dt[x+2,y])
			end
		end
		x = 1
		if img[x,y] == 0
			dt[x,y] = min(dt[x,y], 4 + dt[x+1,y+1], 3 + dt[x,y+1], 3 + dt[x+1,y])
		end
	end
	return dt
end

# ╔═╡ 55416422-d40a-48cf-a6f9-594138f1f53a
ar = [1  1  0  0
 	  0  1  0  0
 	  0  1  0  0
 	  0  1  0  0]

# ╔═╡ 6e5b35b0-3358-425a-bb5e-19ac91e5b8ad
chamfer_distance(ar)

# ╔═╡ 8b5a3a65-07fd-4eb6-90ca-02731754f680
md"""
## Create `3d_chamfer_distance`
"""

# ╔═╡ da978918-07e1-4aa7-afd1-bede26c8d9a2
function chamfer_distance3D(x)
	dt = zeros(Int32, size(x))
	for z in 1:size(x)[3]
		dt[:,:,z] = chamfer_distance(x[:,:,z])
	end
	return dt
end

# ╔═╡ a04b1993-8652-4227-afcc-a2c5d18c84e3
begin
	x1 = [1  1  0  0
		  0  1  0  0
		  0  1  0  0
		  0  1  0  0]
	x2 = [1  1  0  0
		  0  1  0  0
		  0  1  0  0
		  0  1  0  0]
	x3D = cat(x1, x2, dims=3)
end

# ╔═╡ fd20acc6-c0a2-48db-ac97-5d4140148ab9
chamfer_distance3D(x3D)

# ╔═╡ 3c3ece65-2ae1-4b4d-936d-ce6fc01d0d7c
function chamfer_distance_5d(x)
	dt = zeros(size(x))
	for batch in 1:size(x)[5]
		for channel in 1:size(x)[4]
			for z in 1:size(x)[3]
				dt[:, :, z, channel, batch] = chamfer_distance(x[:, :, z, channel, batch])
			end
		end
	end
	return dt
end

# ╔═╡ 813181ae-d1db-47b7-b886-778c0b8507cf
md"""
## Time `chamfer_distance_5D`
"""

# ╔═╡ bc88e99f-1969-4a2d-9955-b883068571a3
@benchmark chamfer_distance_5d($x)

# ╔═╡ Cell order:
# ╟─e4105107-121a-4fd4-9090-764dc4a38129
# ╠═3bec943c-d6d4-11eb-114a-81c372ddbd7c
# ╠═c6e99dfb-00df-4217-aaf6-d34d9dd88391
# ╟─2bb02612-8e85-49f0-a680-7a4acbc24926
# ╠═733bf0aa-aab9-4a76-bca3-b8fb5a15f0ca
# ╠═0fe2e0b6-2812-4daa-a3d1-dd6be6b0176f
# ╠═63ae0c38-d7bb-454d-9771-d788d9fa91f9
# ╠═147b796c-1cf3-4b4b-b819-a348bdd01ef6
# ╠═c6aa66e0-877d-4e85-ae24-3febaffef557
# ╠═c06fc005-f48e-4f9f-8ce8-a88e9c458fe5
# ╟─d4b055c3-c4cb-4a05-9529-a30858916f7f
# ╠═2e3093d8-8a00-49dd-934f-3bd8e6534dab
# ╠═3d657311-428a-456d-a457-4bf314e2c7d7
# ╟─94ffcb2d-7cd4-4b1c-a07a-e2bde42bdd43
# ╠═c92f21b2-5d84-495a-8981-af1a1fbd774e
# ╠═bba80cc8-f9e3-4523-9b7c-25489aa90d41
# ╠═a6107a3f-eecd-4b8c-a298-c9bdf3f5bbec
# ╟─50a6a2d0-5436-446c-9b01-6624a60300ba
# ╠═e86f986d-b0cb-44b4-a0a3-17651037e01c
# ╟─01942fc2-e32b-437a-a3a3-7d2385d5f32a
# ╠═7d5f4bc1-455a-4ebb-b384-cc73c6e796c7
# ╠═55416422-d40a-48cf-a6f9-594138f1f53a
# ╠═6e5b35b0-3358-425a-bb5e-19ac91e5b8ad
# ╟─8b5a3a65-07fd-4eb6-90ca-02731754f680
# ╠═da978918-07e1-4aa7-afd1-bede26c8d9a2
# ╠═a04b1993-8652-4227-afcc-a2c5d18c84e3
# ╠═fd20acc6-c0a2-48db-ac97-5d4140148ab9
# ╠═3c3ece65-2ae1-4b4d-936d-ce6fc01d0d7c
# ╟─813181ae-d1db-47b7-b886-778c0b8507cf
# ╠═bc88e99f-1969-4a2d-9955-b883068571a3
