### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 5c271c90-0d29-11ed-0f2f-5d1c923a6a3d
# ╠═╡ show_logs = false
begin
	let
		using Pkg
		Pkg.activate(mktempdir())
		Pkg.Registry.update()
		Pkg.add("PlutoUI")
		Pkg.add("CairoMakie")
		Pkg.add("Luxor")
		Pkg.add("BenchmarkTools")
		Pkg.add("ColorTypes")
		Pkg.add(url="https://github.com/Dale-Black/ActiveContours.jl")
		Pkg.add(url="https://github.com/Dale-Black/DistanceTransforms.jl")
	end
	
	using PlutoUI
	using CairoMakie
	using LinearAlgebra
	using Luxor
	using BenchmarkTools
	using ColorTypes
	using ActiveContours
	using DistanceTransforms
end

# ╔═╡ d5c043ae-3f29-4209-abb5-23fa65884a7c
TableOfContents()

# ╔═╡ a80aafd1-6ea7-45a0-aa1b-0a9ccae5cd0b
boolean_indicator(f) = @. ifelse(f == 0, 1f10, 0f0)

# ╔═╡ f15e6e16-0bfd-4638-b48d-a976e9b06c10
struct SquaredEuclidean <: DistanceTransforms.DistanceTransform end

# ╔═╡ 3653fd1d-1198-4e92-92f2-ac8b2aa96685
md"""
## 1D
"""

# ╔═╡ 03bd1c7c-b07f-4b5f-b52f-8d96bcbdac58
function transform(f::AbstractVector, tfm::SquaredEuclidean; output=zeros(length(f)), v=ones(Int32, length(f)), z=ones(length(f)))
	z[1] = -Inf32
	z[2] = Inf32
	k = 1; # Index of the rightmost parabola in the lower envelope
	for q = 2:length(f)
		s = ((f[q] + q^2) - (f[v[k]] + v[k]^2)) / (2*q - 2*v[k])
	    while s ≤ z[k]
	        k = k - 1
	        s = ((f[q] + q^2) - (f[v[k]] + v[k]^2)) / (2*q - 2*v[k])
	    end
	    k = k + 1
	    v[k] = q
	    z[k] = s
		if k ≤ length(f) - 1
			z[k+1] = Inf32
		else
			z[k] = Inf32
		end
	end
	k = 1
	for q in 1:length(f)
	    while z[k+1] < q
	        k = k+1
	    end
	    output[q] = (q-v[k])^2 + f[v[k]]
	end
	return output
end

# ╔═╡ bf5eb295-fa19-4249-bb7a-704e255f62d4
md"""
### Tests
"""

# ╔═╡ e0363a94-6efd-493a-bc1c-5c0111094dad
md"""
## 2D
"""

# ╔═╡ c4011956-c903-4191-b1a2-a3315c05c761
function transform(img::AbstractMatrix, tfm::SquaredEuclidean; output=zeros(size(img)), v=ones(Int32, size(img)), z=ones(size(img)))
	for i in axes(img, 1)
	    output[i, :] = transform(img[i, :], tfm; output=output[i,:], z=z[i,:], v=v[i,:])
	end
	for j in axes(img, 2)
	    output[:, j] = transform(output[:, j], tfm; output=output[:,j], z=z[:,j], v=v[:,j])
	end
	return output
end

# ╔═╡ 15e1cb37-7717-425c-8028-2375998f491f
# function DT2(img, tfm::SquaredEuclidean; output=zeros(size(img)), v=ones(Int32, size(img)), z=ones(size(img)))
# 	output = mapslices(x->transform(x, tfm; output=output, v=v, z=z), img, dims = 1)
#     output = mapslices(x->transform(x, tfm; output=output, v=z, z=z), output, dims = 2)
# 	return output
# end

# ╔═╡ 1d0dd515-a341-4f22-9b64-e671534d3167
md"""
### Tests
"""

# ╔═╡ 3a8c65ab-fe56-4679-b25e-cf115829e59d
# let
# 	img = [
# 		0 1 1 1 0 0 0 1 1
# 		1 1 1 1 1 0 0 0 1
# 		1 0 0 0 1 0 0 1 1
# 		1 0 0 0 1 0 1 1 0
# 		1 0 0 0 1 1 0 1 0
# 		1 1 1 1 1 0 0 1 0
# 		0 1 1 1 0 0 0 0 1
# 	]
# 	output = zeros(size(img))
# 	tfm = SquaredEuclidean()
# 	test = DT2(boolean_indicator(img), tfm; output=output)
# end

# ╔═╡ ab548a68-e5bc-4f28-a7a9-b5840a056fe8
# let
# 	img = [
# 		0 1 1 1 0 0 0 1 1
# 		1 1 1 1 1 0 0 0 1
# 		1 0 0 0 1 0 0 1 1
# 		1 0 0 0 1 0 1 1 0
# 		1 0 0 0 1 1 0 1 0
# 		1 1 1 1 1 0 0 1 0
# 		0 1 1 1 0 0 0 0 1
# 	]
# 	output = zeros(size(img))
# 	tfm = SquaredEuclidean()
# 	@benchmark DT2($boolean_indicator($img), $tfm; output=$output)
# end

# ╔═╡ d733ef61-d519-4944-be51-d217b7dc43b6
md"""
## 3D
"""

# ╔═╡ eb7483cd-a12b-4787-b495-6225cd18cba0
function transform(vol::AbstractArray, tfm::SquaredEuclidean;
output=zeros(size(vol)), v=ones(Int32, size(vol)), z=ones(size(vol)))
	for k in axes(vol, 3)
	    output[:, :, k] = transform(boolean_indicator(vol[:, :, k]), tfm; output=output[:, :, k], v=v[:, :, k], z=z[:, :, k])
	end
	for i in axes(vol, 1)
		for j in axes(vol, 2)
	    	output[i, j, :] = transform(output[i, j, :], tfm; output=output[i, j, :], v=v[i, j, :], z=z[i, j, :])
		end
	end
	return output
end

# ╔═╡ 19f5a42a-9366-4032-9394-eaf142186198
let
	f = [1, 1, 0, 0, 0, 1, 1]
	arg1, arg2, arg3 = zeros(length(f)), ones(Int32, length(f)), ones(length(f))
	tfm = SquaredEuclidean()
	test = transform(boolean_indicator(f), tfm; output=arg1, v=arg2, z=arg3)
end

# ╔═╡ e370d94c-3129-420a-91e2-a81fe8aaabc9
let
	f = [0, 0, 0, 1]
	arg1, arg2, arg3 = zeros(length(f)), ones(Int32, length(f)), ones(length(f))
	tfm = SquaredEuclidean()
	test = transform(boolean_indicator(f), tfm; output=arg1, v=arg2, z=arg3)
end

# ╔═╡ d4cad23a-166c-40de-92aa-43f0382597d9
let
	f = [1, 0, 0, 0]
	arg1, arg2, arg3 = zeros(length(f)), ones(Int32, length(f)), ones(length(f))
	tfm = SquaredEuclidean()
	test = transform(boolean_indicator(f), tfm; output=arg1, v=arg2, z=arg3)
end

# ╔═╡ abd91126-4258-4047-8b20-ccc73d5128d3
let
	f = [1, 0, 0, 0]
	arg1, arg2, arg3 = zeros(length(f)), ones(Int32, length(f)), ones(length(f))
	tfm = SquaredEuclidean()
	@benchmark transform($boolean_indicator($f), $tfm; output=$arg1, v=$arg2, z=$arg3)
end

# ╔═╡ a1c564e0-5edc-43fa-975f-bc7103e9c518
let
	img = [
		0 1 1 1 0 0 0 1 1
		1 1 1 1 1 0 0 0 1
		1 0 0 0 1 0 0 1 1
		1 0 0 0 1 0 1 1 0
		1 0 0 0 1 1 0 1 0
		1 1 1 1 1 0 0 1 0
		0 1 1 1 0 0 0 0 1
	]
	output, v, z = zeros(size(img)), ones(Int32, size(img)), ones(size(img))
	tfm = SquaredEuclidean()
	test = transform(boolean_indicator(img), tfm; output=output, v=v, z=z)
end

# ╔═╡ ce3dff0a-589a-482d-8aca-51ba504c7396
let
	img = [
		0 1 1 1 0 0 0 1 1
		1 1 1 1 1 0 0 0 1
		1 0 0 0 1 0 0 1 1
		1 0 0 0 1 0 1 1 0
		1 0 0 0 1 1 0 1 0
		1 1 1 1 1 0 0 1 0
		0 1 1 1 0 0 0 0 1
	]
	output, v, z = zeros(size(img)), ones(Int32, size(img)), ones(size(img))
	tfm = SquaredEuclidean()
	@benchmark transform($boolean_indicator($img), $tfm; output=$output, v=$v, z=$z)
end

# ╔═╡ 3974b7ff-9c85-4b2b-86cb-1907b018f209
let
	img = [
		0 0 0 0 0 0 0 0 0 0 0
		0 0 0 0 0 0 0 0 0 0 0
		0 0 0 0 0 0 0 0	0 0 0
		0 0 0 1 1 1 0 0 0 0 0
		0 0 1 0 0 1 0 0 0 0 0
		0 0 1 0 0 1 1 1 0 0 0
		0 0 1 0 0 0 0 1 0 0 0
		0 0 1 0 0 0 0 1 0 0 0
		0 0 0 1 1 1 1 0 0 0 0	
		0 0 0 0 0 0 0 0 0 0 0	
		0 0 0 0 0 0 0 0 0 0 0
	]
	output, v, z = zeros(size(img)), ones(Int32, size(img)), ones(size(img))
	tfm = SquaredEuclidean()
	test = transform(boolean_indicator(img), tfm; output=output, v=v, z=z)
end

# ╔═╡ f47f5ba9-effd-468f-9ec4-eea9438c30fb
let
	img = [
		0 0 0 0 0 0 0 0 0 0 0
		0 0 0 0 0 0 0 0 0 0 0
		0 0 0 0 0 0 0 0	0 0 0
		0 0 0 1 1 1 0 0 0 0 0
		0 0 1 0 0 1 0 0 0 0 0
		0 0 1 0 0 1 1 1 0 0 0
		0 0 1 0 0 0 0 1 0 0 0
		0 0 1 0 0 0 0 1 0 0 0
		0 0 0 1 1 1 1 0 0 0 0	
		0 0 0 0 0 0 0 0 0 0 0	
		0 0 0 0 0 0 0 0 0 0 0
	]
	output, v, z = zeros(size(img)), ones(Int32, size(img)), ones(size(img))
	tfm = SquaredEuclidean()
	@benchmark transform($boolean_indicator($img), $tfm; output=$output, v=$v, z=$z)
end

# ╔═╡ 3f0c266c-da61-4ba7-a7fb-7dc171df19f9
md"""
### Tests
"""

# ╔═╡ 53e2854f-c805-46ad-a133-7b6988eab426
let
	img = [
		0 0 0 0 0 0 0 0 0 0 0
		0 0 0 0 0 0 0 0 0 0 0
		0 0 0 0 0 0 0 0	0 0 0
		0 0 0 1 1 1 0 0 0 0 0
		0 0 1 0 0 1 0 0 0 0 0
		0 0 1 0 0 1 1 1 0 0 0
		0 0 1 0 0 0 0 1 0 0 0
		0 0 1 0 0 0 0 1 0 0 0
		0 0 0 1 1 1 1 0 0 0 0	
		0 0 0 0 0 0 0 0 0 0 0	
		0 0 0 0 0 0 0 0 0 0 0
	]
	img_inv = @. ifelse(img == 0, 1, 0)
	vol = cat(img, img_inv, dims=3)
	container2 = []
	for i in 1:10
		push!(container2, vol)
	end
	vol_inv = cat(container2..., dims=3)
	output, v, z = zeros(size(vol_inv)), ones(Int32, size(vol_inv)), ones(size(vol_inv))
	tfm = SquaredEuclidean()
	test = transform(boolean_indicator(vol_inv), tfm; output=output, v=v, z=z)
end

# ╔═╡ 5d309f7a-3805-416d-b1b1-d7f9c0d40eac
let
	img = [
		0 0 0 0 0 0 0 0 0 0 0
		0 0 0 0 0 0 0 0 0 0 0
		0 0 0 0 0 0 0 0	0 0 0
		0 0 0 1 1 1 0 0 0 0 0
		0 0 1 0 0 1 0 0 0 0 0
		0 0 1 0 0 1 1 1 0 0 0
		0 0 1 0 0 0 0 1 0 0 0
		0 0 1 0 0 0 0 1 0 0 0
		0 0 0 1 1 1 1 0 0 0 0	
		0 0 0 0 0 0 0 0 0 0 0	
		0 0 0 0 0 0 0 0 0 0 0
	]
	img_inv = @. ifelse(img == 0, 1, 0)
	vol = cat(img, img_inv, dims=3)
	container2 = []
	for i in 1:10
		push!(container2, vol)
	end
	vol_inv = cat(container2..., dims=3)
	output, v, z = zeros(size(vol_inv)), ones(Int32, size(vol_inv)), ones(size(vol_inv))
	tfm = SquaredEuclidean()
	@benchmark transform($boolean_indicator($vol_inv), $tfm; output=$output, v=$v, z=$z)
end

# ╔═╡ 54bbefa8-cbc5-4c3c-a8aa-e54d603b552e
begin
	img = [
		0 0 0 0 0 0 0 0 0 0 0
		0 0 0 0 0 0 0 0 0 0 0
		0 0 0 0 0 0 0 0	0 0 0
		0 0 0 1 1 1 0 0 0 0 0
		0 0 1 0 0 1 0 0 0 0 0
		0 0 1 0 0 1 1 1 0 0 0
		0 0 1 0 0 0 0 1 0 0 0
		0 0 1 0 0 0 0 1 0 0 0
		0 0 0 1 1 1 1 0 0 0 0	
		0 0 0 0 0 0 0 0 0 0 0	
		0 0 0 0 0 0 0 0 0 0 0
	]
	img_inv = @. ifelse(img == 0, 1, 0)
	a1 = img_inv
	a2 = img
	ans = cat(a1, a2, dims=3)
	container_a = []
	for i in 1:10
		push!(container_a, ans)
	end
	answer = cat(container_a..., dims=3)
end

# ╔═╡ Cell order:
# ╠═5c271c90-0d29-11ed-0f2f-5d1c923a6a3d
# ╠═d5c043ae-3f29-4209-abb5-23fa65884a7c
# ╠═a80aafd1-6ea7-45a0-aa1b-0a9ccae5cd0b
# ╠═f15e6e16-0bfd-4638-b48d-a976e9b06c10
# ╟─3653fd1d-1198-4e92-92f2-ac8b2aa96685
# ╠═03bd1c7c-b07f-4b5f-b52f-8d96bcbdac58
# ╟─bf5eb295-fa19-4249-bb7a-704e255f62d4
# ╠═19f5a42a-9366-4032-9394-eaf142186198
# ╠═e370d94c-3129-420a-91e2-a81fe8aaabc9
# ╠═d4cad23a-166c-40de-92aa-43f0382597d9
# ╠═abd91126-4258-4047-8b20-ccc73d5128d3
# ╟─e0363a94-6efd-493a-bc1c-5c0111094dad
# ╠═c4011956-c903-4191-b1a2-a3315c05c761
# ╟─15e1cb37-7717-425c-8028-2375998f491f
# ╟─1d0dd515-a341-4f22-9b64-e671534d3167
# ╠═a1c564e0-5edc-43fa-975f-bc7103e9c518
# ╠═ce3dff0a-589a-482d-8aca-51ba504c7396
# ╠═3974b7ff-9c85-4b2b-86cb-1907b018f209
# ╠═f47f5ba9-effd-468f-9ec4-eea9438c30fb
# ╟─3a8c65ab-fe56-4679-b25e-cf115829e59d
# ╟─ab548a68-e5bc-4f28-a7a9-b5840a056fe8
# ╟─d733ef61-d519-4944-be51-d217b7dc43b6
# ╠═eb7483cd-a12b-4787-b495-6225cd18cba0
# ╟─3f0c266c-da61-4ba7-a7fb-7dc171df19f9
# ╠═53e2854f-c805-46ad-a133-7b6988eab426
# ╠═5d309f7a-3805-416d-b1b1-d7f9c0d40eac
# ╠═54bbefa8-cbc5-4c3c-a8aa-e54d603b552e