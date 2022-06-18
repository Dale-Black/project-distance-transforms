### A Pluto.jl notebook ###
# v0.19.8

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ aec0d43c-16e0-4092-b81e-6dddbe41d3db
# ╠═╡ show_logs = false
begin
	let
		using Pkg
		Pkg.activate(raw"C:\Users\wenbl13\Desktop\wenbo branch\project-distance-transforms\dependency")
		# Pkg.instantiate()
		# Pkg.Registry.update()
		# Pkg.add("PlutoUI")
		# Pkg.add("CairoMakie")
		# Pkg.add("Luxor")
		# Pkg.add("BenchmarkTools")
		# Pkg.add("ColorTypes")
		# Pkg.add(url="https://github.com/Dale-Black/ActiveContours.jl")
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
boolean_indicator(f) = @. ifelse(f == 0, 1f10, 0f0)

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

# ╔═╡ e5b89de2-64de-42d1-92d8-001a7e7d1bff
begin
	function DT1WenboJune16(f; output=zeros(length(f)), pointerA=1, pointerB=1)
		#assume length(f)>0
		#This is a one pass algorithm
		#time complexity=O(n), Space complexity=O(1)
		while (pointerA<=length(f))
			if(f[pointerA] == 0)
				output[pointerA]=0
				pointerA=pointerA+1
				pointerB=pointerB+1
			else
				while(pointerB <= length(f) && f[pointerB]==1f10)
					pointerB=pointerB+1
				end
				if (pointerB > length(f))
					if (pointerA == 1)
						output = DT1helper(f, output, -1, -1)
					else
						output = DT1helper(f, output, pointerA, -1)
					end
				else
					if (pointerA == 1)
						output = DT1helper(f, output, -1, pointerB-1)
					else
						output = DT1helper(f, output, pointerA, pointerB-1)
					end
				end
				pointerA=pointerB
			end
		end
		return output
	end
	
	function DT1helper(input, output, i, j)
		if (i==-1 && j==-1)
			i=1
			while(i<=length(output))
				output[i]=1f10
				i=i+1
			end
		elseif(i==-1)
			temp=1
			while(j>0)
				output[j]=temp^2
				j=j-1
				temp=temp+1
			end
		elseif(j==-1)
			temp=1
			while(i<=length(output))
				output[i]=temp^2
				i=i+1
				temp=temp+1
			end
		else
			temp=1
			while(i<=j)
				output[i]=output[j]=temp^2
				temp=temp+1
				i=i+1
				j=j-1
			end
		end
		return output
	end
end

# ╔═╡ 43556a83-fac3-444f-9bfa-96a3b9d9bf8e
begin
	function DT1ForDT2(f; output=zeros(length(f)), pointerA=1, pointerB=1)
		#assume length(f)>0
		#This is a one pass algorithm
		#time complexity=O(n), Space complexity=O(1)
		while (pointerA<=length(f))
			if(f[pointerA] != 1f10)
				output[pointerA]=f[pointerA]
				pointerA=pointerA+1
				pointerB=pointerB+1
			else
				while(pointerB <= length(f) && f[pointerB]==1f10)
					pointerB=pointerB+1
				end
				if (pointerB > length(f))
					if (pointerA == 1)
						output = DT1ForDT2Helper(f, output, -1, -1)
					else
						output = DT1ForDT2Helper(f, output, pointerA, -1)
					end
				else
					if (pointerA == 1)
						output = DT1ForDT2Helper(f, output, -1, pointerB-1)
					else
						output = DT1ForDT2Helper(f, output, pointerA, pointerB-1)
					end
				end
				pointerA=pointerB
			end
		end
		return output
	end
	
	function DT1ForDT2Helper(input, output, i, j)
		if (i==-1 && j==-1)
			i=1
			while(i<=length(output))
				output[i]=min(input[i],1f10)
				i=i+1
			end
		elseif(i==-1)
			temp=1
			i=j+1
			while(j>0)
				output[j]=min(input[j],input[i]+temp^2)
				j=j-1
				temp=temp+1
			end
		elseif(j==-1)
			temp=1
			j=i-1
			while(i<=length(output))
				output[i]=min(input[i],input[j]+temp^2)
				i=i+1
				temp=temp+1
			end
		else
			temp=1
			residueA=input[i-1]
			residueB=input[j+1]
			while(i<=j)
				output[i]=min(input[i],residueA+temp^2)
				output[j]=min(input[j],residueB+temp^2)
				temp=temp+1
				i=i+1
				j=j-1
			end
		end
		return output
	end
end

# ╔═╡ 33cf18e1-4751-44fc-8c2d-eafaeb21aa44
DT1WenboJune16(boolean_indicator([0 1 1 0 0 0 0 1 1 0 0 0 1 0]))
# DT1WenboJune16(boolean_indicator([0 0 0 0]))

# ╔═╡ 8ea2604c-dd48-42ef-ae0d-414f84217691
DT1([1f10 1f10 1f10 0])

# ╔═╡ fa5836cb-6074-4c36-81e7-0995acce01b2
DT1WenboJune16([1f10 1f10 1f10 0])

# ╔═╡ 2280361b-068d-4d5e-ae30-7e66c42ee665
function testDT1Wenbo(size,numCases)
	for i in 1:numCases
		testInput=rand(0:1,1,size);
		if (DT1(boolean_indicator(testInput)) != DT1WenboJune16(boolean_indicator(testInput)))
			println(testInput);
			println(DT1(boolean_indicator(testInput)));
			println(DT1WenboJune16(boolean_indicator(testInput)));
			println();
			return "Failed."
		end 
	end
	return "Passed."		
end

# ╔═╡ c8e739c0-940d-47b4-be2e-b0345a7e362f
testDT1Wenbo(5,5000)

# ╔═╡ 0469116a-7e10-44c5-801d-f5ce683dc486
f1 = [1 1 0 0 0 1 1];

# ╔═╡ 72f886e7-f0b8-46f2-9190-91d984abef4c
D, v, z =zeros(length(f1)), ones(Int32, length(f1)), ones(length(f1))

# ╔═╡ 30ae8059-4b4e-4026-966e-7d6320912b5c
@benchmark DT1($boolean_indicator($f1); D=$D, v=$v, z=$z) 

# ╔═╡ b44d44c5-eaac-4eb2-ba44-8346f55fb371
arg1, arg2, arg3 = zeros(length(f1)), 1, 1

# ╔═╡ 7a1a556f-fa9e-4cec-88e8-5c74c10e0c4e
@benchmark DT1WenboJune16($boolean_indicator($f1); output=$arg1, pointerA=$arg2, pointerB=$arg3) 

# ╔═╡ 2d02a6fd-7a23-4790-b8da-7b7150482a85
md"""
## 2D
"""

# ╔═╡ 29cb3c08-a702-4564-ad41-664f312913b0
x1 = [
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

# ╔═╡ 46863913-57e1-47cf-86b3-1933c9e45bc7
img = [
	 0 1 1 1 0 0 0 1 1
	 1 1 1 1 1 0 0 0 1
	 1 0 0 0 1 0 0 1 1
	 1 0 0 0 1 0 1 1 0
	 1 0 0 0 1 1 0 1 0
	 1 1 1 1 1 0 0 1 0
	 0 1 1 1 0 0 0 0 1
]



# ╔═╡ e3851249-ddf2-46e4-8805-d01da6ba0d4b
# img = [
# 	 1.0e10  0.0     0.0     0.0     1.0e10  1.0e10  1.0e10  0.0     0.0
# 	 0.0     0.0     0.0     0.0     0.0     1.0e10  1.0e10  1.0e10  0.0
# 	 0.0     1.0e10  1.0e10  1.0e10  0.0     1.0e10  1.0e10  0.0     0.0
# 	 0.0     1.0e10  1.0e10  1.0e10  0.0     1.0e10  0.0     0.0     1.0e10
# 	 0.0     1.0e10  1.0e10  1.0e10  0.0     0.0     1.0e10  0.0     1.0e10
# 	 0.0     0.0     0.0     0.0     0.0     1.0e10  1.0e10  0.0     1.0e10
# 	 1.0e10  0.0     0.0     0.0     1.0e10  1.0e10  1.0e10  1.0e10  0.0
# ]

# ╔═╡ b3ba99a1-cbc9-484f-93a5-7a235905cbf8
function DT20(img; D=zeros(size(img)))
	for i = 1:size(img, 1)
	    D[i, :] = DT1WenboJune16(img[i, :])
	end
	return D
end

# ╔═╡ 2d7da285-7078-4567-8cdd-a8cefd737f57
function DT2(img; D=zeros(size(img)))
	for i = 1:size(img, 1)
	    D[i, :] = DT1WenboJune16(img[i, :])
	end
	for j = 1:size(img, 2)
	    D[:, j] = DT1ForDT2(D[:, j])
	end
	return D
end

# ╔═╡ c7129d90-0d23-4798-8e0f-5eef0b017533
function DT22(img; D=zeros(size(img)))
	D = mapslices(DT1WenboJune16, img, dims = 2)
    D = mapslices(DT1ForDT2, D, dims = 1)
	return D
end

# ╔═╡ d751a3f7-caea-4d76-8565-3da6757bd8f1
DT20(boolean_indicator(x1))

# ╔═╡ f7402111-bf79-476e-8504-689fed5a08ec
DT2(boolean_indicator(x1))

# ╔═╡ ad3fff84-0c1f-491c-8dd6-95069f4b6e02
img2 = [
	0 0 0 0
	0 0 1 0
	0 1 1 0
	0 0 0 0
]

# ╔═╡ 74ce6426-6c76-43f7-ab43-fcbb5f1015de
DT2(boolean_indicator(img2))

# ╔═╡ 538619d7-8c4d-47c4-bf5a-88165ea26ac4
d0 = DT2(boolean_indicator(img))

# ╔═╡ a60641bc-cfc9-4963-a363-f07387bbc879
d2 = DT22(boolean_indicator(img))

# ╔═╡ 8a5666f7-ee15-4f8c-847f-9fb998a89c7e
# @benchmark DT21($img)

# ╔═╡ ef8cde04-6e7c-4b2a-864b-e3fee501d6a3
# @benchmark DT22($img)

# ╔═╡ 71072c67-6584-4fc7-ad4d-b3d362333562
md"""
## 3D
"""

# ╔═╡ 740b8c8a-9551-4643-b8c9-aa09b9bd9ae7
# function DT3(vol)
#     D = mapslices(DT1, vol, dims = 2)
#     D = mapslices(DT1, D, dims = 1)
#     return mapslices(DT1, D, dims = 3)
# end

# ╔═╡ 0fe6bcfc-f05b-4e98-a0a1-e2652657e46d
# begin
# 	img_inv = @. ifelse(img == 0, 1e10, 0)
# 	vol = cat(img, img_inv, dims=3)
# 	container = []
# 	for i in 1:10
# 		push!(container, vol)
# 	end
# 	vol_new = cat(container..., dims=3)
# end;

# ╔═╡ ac8ed96d-bf48-4a2e-81f0-ae612024177c
# @bind c PlutoUI.Slider(1:size(vol_new, 3), default=1, show_value=true)

# ╔═╡ fad6db76-5bd3-4350-8180-eb9735f6a749
# heatmap(vol_new[:, :, c])

# ╔═╡ 33ce5963-4fb7-44e8-9206-d4f6d828db97
# dt_vol_3d = DT3(vol_new)

# ╔═╡ ab59045a-448e-4d09-b779-be372b7b9793
# @bind b PlutoUI.Slider(1:size(dt_vol_3d, 3), default=1, show_value=true)

# ╔═╡ 66c52751-0b74-4204-9eba-71d577c0ecd0
# heatmap(dt_vol_3d[:, :, b])

# ╔═╡ b4d5b33a-4126-46d4-9128-ce778ccbfed8
md"""
## Inverse
"""

# ╔═╡ 8e8512c5-e547-4a67-9ec0-936bbe6e58ad
md"""
In 3D, if you stack 2D arrays along with their inverses in a repeating way, the distance transform should result in only zeros and ones. This is because the background element of a 2D slice is always going to be adjacent to a foreground element of the adjacent 2D slice since the adjacent slice is simply the inverted array. This is a good way to test if the 3D form of our distance transform is working or not
"""

# ╔═╡ 65bcf599-e7f2-49dc-a0ec-b977ded03efe
# begin
# 	vol_inv = cat(img, img_inv, dims=3)
# 	vol_inv = cat(vol_inv, vol_inv, dims=3)
# end

# ╔═╡ 84bfe423-b20d-4e2d-a31e-281755742b0c
# dt_vol_inv = DT3(vol_inv)

# ╔═╡ 172c73bc-39c1-459f-9e56-b716fc798361
# @bind a PlutoUI.Slider(1:size(dt_vol_inv, 3), default=1, show_value=true)

# ╔═╡ 0a6ab1a0-98b8-4e24-8ad9-eae7dd6f42a7
# heatmap(dt_vol_inv[:, :, a])

# ╔═╡ Cell order:
# ╠═aec0d43c-16e0-4092-b81e-6dddbe41d3db
# ╠═45c7f27a-b43c-4781-918a-51aebd273014
# ╠═45b90a66-956a-4b35-9e88-7621fd5c9c1a
# ╟─8aab09c7-07fe-4691-9e1d-35e2e215bc5d
# ╟─e75c9e76-3634-4bde-abb8-5b4827eb089e
# ╟─17d3777b-69d4-40c1-84c1-2bc6a737d52a
# ╠═258346da-e792-11ec-06dc-5906e95d24e2
# ╠═e5b89de2-64de-42d1-92d8-001a7e7d1bff
# ╠═43556a83-fac3-444f-9bfa-96a3b9d9bf8e
# ╠═33cf18e1-4751-44fc-8c2d-eafaeb21aa44
# ╠═8ea2604c-dd48-42ef-ae0d-414f84217691
# ╠═fa5836cb-6074-4c36-81e7-0995acce01b2
# ╠═2280361b-068d-4d5e-ae30-7e66c42ee665
# ╠═c8e739c0-940d-47b4-be2e-b0345a7e362f
# ╠═0469116a-7e10-44c5-801d-f5ce683dc486
# ╠═72f886e7-f0b8-46f2-9190-91d984abef4c
# ╠═30ae8059-4b4e-4026-966e-7d6320912b5c
# ╠═b44d44c5-eaac-4eb2-ba44-8346f55fb371
# ╠═7a1a556f-fa9e-4cec-88e8-5c74c10e0c4e
# ╟─2d02a6fd-7a23-4790-b8da-7b7150482a85
# ╠═29cb3c08-a702-4564-ad41-664f312913b0
# ╠═46863913-57e1-47cf-86b3-1933c9e45bc7
# ╠═e3851249-ddf2-46e4-8805-d01da6ba0d4b
# ╠═b3ba99a1-cbc9-484f-93a5-7a235905cbf8
# ╠═2d7da285-7078-4567-8cdd-a8cefd737f57
# ╠═c7129d90-0d23-4798-8e0f-5eef0b017533
# ╠═d751a3f7-caea-4d76-8565-3da6757bd8f1
# ╠═f7402111-bf79-476e-8504-689fed5a08ec
# ╠═ad3fff84-0c1f-491c-8dd6-95069f4b6e02
# ╠═74ce6426-6c76-43f7-ab43-fcbb5f1015de
# ╠═538619d7-8c4d-47c4-bf5a-88165ea26ac4
# ╠═a60641bc-cfc9-4963-a363-f07387bbc879
# ╠═8a5666f7-ee15-4f8c-847f-9fb998a89c7e
# ╠═ef8cde04-6e7c-4b2a-864b-e3fee501d6a3
# ╟─71072c67-6584-4fc7-ad4d-b3d362333562
# ╠═740b8c8a-9551-4643-b8c9-aa09b9bd9ae7
# ╠═0fe6bcfc-f05b-4e98-a0a1-e2652657e46d
# ╠═ac8ed96d-bf48-4a2e-81f0-ae612024177c
# ╠═fad6db76-5bd3-4350-8180-eb9735f6a749
# ╠═33ce5963-4fb7-44e8-9206-d4f6d828db97
# ╠═ab59045a-448e-4d09-b779-be372b7b9793
# ╠═66c52751-0b74-4204-9eba-71d577c0ecd0
# ╟─b4d5b33a-4126-46d4-9128-ce778ccbfed8
# ╟─8e8512c5-e547-4a67-9ec0-936bbe6e58ad
# ╠═65bcf599-e7f2-49dc-a0ec-b977ded03efe
# ╠═84bfe423-b20d-4e2d-a31e-281755742b0c
# ╠═172c73bc-39c1-459f-9e56-b716fc798361
# ╠═0a6ab1a0-98b8-4e24-8ad9-eae7dd6f42a7
