### A Pluto.jl notebook ###
# v0.19.8

using Markdown
using InteractiveUtils

# ╔═╡ aec0d43c-16e0-4092-b81e-6dddbe41d3db
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
	end
	
	using PlutoUI
	using CairoMakie
	using LinearAlgebra
	using Luxor
	using BenchmarkTools
	using ColorTypes
	using ActiveContours
end

# ╔═╡ 45c7f27a-b43c-4781-918a-51aebd273014
TableOfContents()

# ╔═╡ 45b90a66-956a-4b35-9e88-7621fd5c9c1a
boolean_indicator(f) = @. ifelse(f == 0, 1f12, 0f0)

# ╔═╡ 8aab09c7-07fe-4691-9e1d-35e2e215bc5d
md"""
## TODO
- Make sure the 3D distance transform is working
- Make sure the distance transforms (1D, 2D, 3D) all have a working non-allocating version (use BenchmarkTools and a profiler to validate this)
"""

# ╔═╡ e75c9e76-3634-4bde-abb8-5b4827eb089e
md"""
## 1D
"""

# ╔═╡ 17d3777b-69d4-40c1-84c1-2bc6a737d52a
"""
One-dimensional generalized distance transform
Input: f - the sampled function
Output: D - distance transform

Based on the paper:
P. Felzenszwalb, D. Huttenlocher
Distance Transforms of Sampled Functions
Cornell Computing and Information Science Technical Report TR2004-1963, September 2004
"""

# ╔═╡ 258346da-e792-11ec-06dc-5906e95d24e2
function DT1(f; D=zeros(length(f)), v=ones(Int32, length(f)), z=ones(length(f)))
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
	    D[q] = (q-v[k])^2 + f[v[k]]
	end
	return D
end

# ╔═╡ fdce0941-d69c-474e-9e7c-8ecf8dadaeb5
DT1(boolean_indicator([1,1,0]))

# ╔═╡ 2f6d520b-57e2-44ec-8c40-ee24bbeda983
DT1(boolean_indicator([1, 0, 1]))

# ╔═╡ 0e7291fc-d326-4e96-8f4e-0c868ce37702
DT1(boolean_indicator([0, 1, 0]))

# ╔═╡ b489ce5a-9b09-4778-8dda-cda5a5875542
DT1(boolean_indicator([1,0,0,0,0,0,0]))

# ╔═╡ 0469116a-7e10-44c5-801d-f5ce683dc486
f1 = [1 0 1 1 1 0 0];

# ╔═╡ 72f886e7-f0b8-46f2-9190-91d984abef4c
D, v, z =zeros(length(f1)), ones(Int32, length(f1)), ones(length(f1))

# ╔═╡ 273a6346-856a-4581-af6d-8c12b4e24069
DT1(boolean_indicator(f1))

# ╔═╡ 30ae8059-4b4e-4026-966e-7d6320912b5c
@benchmark DT1($boolean_indicator($f1); D=$D, v=$v, z=$z) 

# ╔═╡ 2d02a6fd-7a23-4790-b8da-7b7150482a85
md"""
## 2D
"""

# ╔═╡ 29cb3c08-a702-4564-ad41-664f312913b0
x1 = [
		0 1 1 1 0
		1 1 1 1 1
		1 0 0 0 1
		1 0 0 0 1
		1 0 0 0 1
		1 1 1 1 1
		0 1 1 1 0
		]

# ╔═╡ 88d3f48a-e886-4b65-8de3-ec49f5ec57ad
boolean_indicator(x1)

# ╔═╡ e3851249-ddf2-46e4-8805-d01da6ba0d4b
img = [
	 1.0e10  0.0     0.0     0.0     1.0e10  1.0e10  1.0e10  0.0     0.0
	 0.0     0.0     0.0     0.0     0.0     1.0e10  1.0e10  1.0e10  0.0
	 0.0     1.0e10  1.0e10  1.0e10  0.0     1.0e10  1.0e10  0.0     0.0
	 0.0     1.0e10  1.0e10  1.0e10  0.0     1.0e10  0.0     0.0     1.0e10
	 0.0     1.0e10  1.0e10  1.0e10  0.0     0.0     1.0e10  0.0     1.0e10
	 0.0     0.0     0.0     0.0     0.0     1.0e10  1.0e10  0.0     1.0e10
	 1.0e10  0.0     0.0     0.0     1.0e10  1.0e10  1.0e10  1.0e10  0.0
]

# ╔═╡ 4867bf5b-ce99-464a-a758-7dbc9bd7e273
function DT2(img; D=zeros(size(img)))
	for i = 1:size(img, 1)
	    d = DT1(img[i, :])
	    D[i, :] = d
	end
	for j = 1:size(img, 2)
	    d = DT1(D[:, j])
	    D[:, j] = d
	end
	return D
end

# ╔═╡ c7129d90-0d23-4798-8e0f-5eef0b017533
function DT22(img; D=zeros(size(img)))
	D = mapslices(DT1, img, dims = 1)
    D = mapslices(DT1, D, dims = 2)
	return D
end

# ╔═╡ 538619d7-8c4d-47c4-bf5a-88165ea26ac4
d = DT2(img)

# ╔═╡ a60641bc-cfc9-4963-a363-f07387bbc879
d2 = DT22(img)

# ╔═╡ 8a5666f7-ee15-4f8c-847f-9fb998a89c7e
@benchmark DT2($img)

# ╔═╡ 71072c67-6584-4fc7-ad4d-b3d362333562
md"""
## 3D
"""

# ╔═╡ 740b8c8a-9551-4643-b8c9-aa09b9bd9ae7
function DT3(vol)
    D = mapslices(DT1, vol, dims = 2)
    D = mapslices(DT1, D, dims = 1)
    return mapslices(DT1, D, dims = 3)
end

# ╔═╡ 0fe6bcfc-f05b-4e98-a0a1-e2652657e46d
begin
	img_inv = @. ifelse(img == 0, 1e10, 0)
	vol = cat(img, img_inv, dims=3)
	container = []
	for i in 1:10
		push!(container, vol)
	end
	vol_new = cat(container..., dims=3)
end;

# ╔═╡ 33ce5963-4fb7-44e8-9206-d4f6d828db97
DT3(vol_new)

# ╔═╡ b4d5b33a-4126-46d4-9128-ce778ccbfed8
md"""
## Inverse
"""

# ╔═╡ 8e8512c5-e547-4a67-9ec0-936bbe6e58ad
md"""
In 3D, if you stack 2D arrays along with their inverses in a repeating way, the distance transform should result in only zeros and ones. This is because the background element of a 2D slice is always going to be adjacent to a foreground element of the adjacent 2D slice since the adjacent slice is simply the inverted array. This is a good way to test if the 3D form of our distance transform is working or not
"""

# ╔═╡ 65bcf599-e7f2-49dc-a0ec-b977ded03efe
begin
	vol_inv = cat(img, img_inv, dims=3)
	vol_inv = cat(vol_inv, vol_inv, dims=3)
end

# ╔═╡ 84bfe423-b20d-4e2d-a31e-281755742b0c
dt_vol_inv = DT3(vol_inv)

# ╔═╡ 0a6ab1a0-98b8-4e24-8ad9-eae7dd6f42a7
heatmap(dt_vol_inv[:, :, 3])

# ╔═╡ Cell order:
# ╠═aec0d43c-16e0-4092-b81e-6dddbe41d3db
# ╠═45c7f27a-b43c-4781-918a-51aebd273014
# ╠═45b90a66-956a-4b35-9e88-7621fd5c9c1a
# ╟─8aab09c7-07fe-4691-9e1d-35e2e215bc5d
# ╟─e75c9e76-3634-4bde-abb8-5b4827eb089e
# ╟─17d3777b-69d4-40c1-84c1-2bc6a737d52a
# ╠═258346da-e792-11ec-06dc-5906e95d24e2
# ╠═fdce0941-d69c-474e-9e7c-8ecf8dadaeb5
# ╠═2f6d520b-57e2-44ec-8c40-ee24bbeda983
# ╠═0e7291fc-d326-4e96-8f4e-0c868ce37702
# ╠═b489ce5a-9b09-4778-8dda-cda5a5875542
# ╠═0469116a-7e10-44c5-801d-f5ce683dc486
# ╠═72f886e7-f0b8-46f2-9190-91d984abef4c
# ╠═273a6346-856a-4581-af6d-8c12b4e24069
# ╠═30ae8059-4b4e-4026-966e-7d6320912b5c
# ╟─2d02a6fd-7a23-4790-b8da-7b7150482a85
# ╠═29cb3c08-a702-4564-ad41-664f312913b0
# ╠═88d3f48a-e886-4b65-8de3-ec49f5ec57ad
# ╠═e3851249-ddf2-46e4-8805-d01da6ba0d4b
# ╠═4867bf5b-ce99-464a-a758-7dbc9bd7e273
# ╠═c7129d90-0d23-4798-8e0f-5eef0b017533
# ╠═538619d7-8c4d-47c4-bf5a-88165ea26ac4
# ╠═a60641bc-cfc9-4963-a363-f07387bbc879
# ╠═8a5666f7-ee15-4f8c-847f-9fb998a89c7e
# ╟─71072c67-6584-4fc7-ad4d-b3d362333562
# ╠═740b8c8a-9551-4643-b8c9-aa09b9bd9ae7
# ╠═0fe6bcfc-f05b-4e98-a0a1-e2652657e46d
# ╠═33ce5963-4fb7-44e8-9206-d4f6d828db97
# ╟─b4d5b33a-4126-46d4-9128-ce778ccbfed8
# ╟─8e8512c5-e547-4a67-9ec0-936bbe6e58ad
# ╠═65bcf599-e7f2-49dc-a0ec-b977ded03efe
# ╠═84bfe423-b20d-4e2d-a31e-281755742b0c
# ╠═0a6ab1a0-98b8-4e24-8ad9-eae7dd6f42a7
