### A Pluto.jl notebook ###
# v0.19.9

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

# ╔═╡ 45c7f27a-b43c-4781-918a-51aebd273014
TableOfContents()

# ╔═╡ 45b90a66-956a-4b35-9e88-7621fd5c9c1a
boolean_indicator(f) = @. ifelse(f == 0, 1f10, 0f0)

# ╔═╡ 9614e200-5b49-4747-be88-2b50becc048f
struct Wenbo <: DistanceTransforms.DistanceTransform end

# ╔═╡ e75c9e76-3634-4bde-abb8-5b4827eb089e
md"""
## 1D
"""

# ╔═╡ e5b89de2-64de-42d1-92d8-001a7e7d1bff
"""
    function _DT1(input, output, i, j)

Helper function for 1-D Wenbo distance transform `transform(f::AbstractVector, tfm::Wenbo)`
"""
function _DT1(output, i, j)
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

# ╔═╡ e5435c7d-5f4d-4698-936f-3ae9906fa9df
"""
    transform(f, tfm::Wenbo; output=zeros(length(f)), pointerA=1, pointerB=1)

Assume length(f)>0. This is a one pass algorithm. Time complexity=O(n). Space complexity=O(1)
"""
function transform(f::AbstractVector, tfm::Wenbo; output=zeros(length(f)), pointerA=1, pointerB=1)
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
					output = _DT1(output, -1, -1)
				else
					output = _DT1(output, pointerA, -1)
				end
			else
				if (pointerA == 1)
					output = _DT1(output, -1, pointerB-1)
				else
					output = _DT1(output, pointerA, pointerB-1)
				end
			end
			pointerA=pointerB
		end
	end
	return output
end

# ╔═╡ ceda014b-5b22-4b76-beb0-766a1b138053
md"""
### Tests
"""

# ╔═╡ 2d02a6fd-7a23-4790-b8da-7b7150482a85
md"""
## 2D
"""

# ╔═╡ b3ba99a1-cbc9-484f-93a5-7a235905cbf8
"""
    _DT2(f; output=zeros(length(f)), pointerA=1)

Helper function for 2-D Wenbo distance transform `transform(f::AbstractVector, tfm::Wenbo)`
Computes the vertical operation.
"""
function _DT2(f; output=zeros(length(f)), pointerA=1)
	while (pointerA<=length(f))
		output[pointerA]=f[pointerA]
		if(f[pointerA] > 1)
			if (length(f) - pointerA <= pointerA - 1)
				temp = 1
				while (output[pointerA]>1 && temp <= length(f) - pointerA)
					if (f[pointerA+temp]<output[pointerA])
						output[pointerA]=min(output[pointerA], f[pointerA+temp]+temp^2)
					end
					if (f[pointerA-temp]<output[pointerA])
						output[pointerA]=min(output[pointerA], f[pointerA-temp]+temp^2)
					end
					temp = temp + 1
				end
				if(f[pointerA] > 1)
					while (output[pointerA]>1 && temp <= pointerA - 1)
						if (f[pointerA-temp]<output[pointerA])
							output[pointerA]=min(output[pointerA], f[pointerA-temp]+temp^2)
						end
						temp = temp + 1
					end
				end
			else
				temp = 1
				while (output[pointerA]>1 && temp <= pointerA - 1)
					if (f[pointerA+temp]<output[pointerA])
						output[pointerA]=min(output[pointerA], f[pointerA+temp]+temp^2)
					end
					if (f[pointerA-temp]<output[pointerA])
						output[pointerA]=min(output[pointerA], f[pointerA-temp]+temp^2)
					end
					temp = temp + 1
				end
				if(f[pointerA] > 1)
					while (output[pointerA]>1 && temp <= length(f) - pointerA)
						if (f[pointerA+temp]<output[pointerA])
							output[pointerA]=min(output[pointerA], f[pointerA+temp]+temp^2)
						end
						temp = temp + 1
					end
				end
			end
		end
		pointerA=pointerA+1
	end
	return output
end

# ╔═╡ 535c4318-d29e-4640-9105-66b16d13bf88
"""
    transform(img::AbstractMatrix, tfm::Wenbo; output=zeros(size(img)), pointerA=1, pointerB=1)

2-D Wenbo Distance Transform.
"""
function transform(img::AbstractMatrix, tfm::Wenbo; output=zeros(size(img)), pointerA=1, pointerB=1)
	# This is a worst case = O(n^3) implementation
	for i in axes(img, 1)
	    output[i, :] = transform(img[i, :], Wenbo(); output=output[i, :], pointerA=pointerA, pointerB=pointerB) 
	end

	for j in axes(img, 2)
	    output[:, j] = _DT2(output[:, j]; output=output[:, j], pointerA=pointerA) 
	end
	return output
end


# ╔═╡ fdc77c42-616a-4822-af8c-1f02bbd8b4a4
md"""
### Tests
"""

# ╔═╡ 71072c67-6584-4fc7-ad4d-b3d362333562
md"""
## 3D
"""

# ╔═╡ f5ff25be-94d2-4e97-bb4d-df36abb4c0a8
"""
    transform(f::AbstractArray, tfm::Wenbo; D=zeros(size(f)), pointerA=1, pointerB=1)

3-D Wenbo Distance Transform.
"""
function transform(f::AbstractArray, tfm::Wenbo; output=zeros(size(f)), pointerA=1, pointerB=1)
	for i in axes(f, 3)
	    output[:, :, i] = transform(f[:, :, i], Wenbo(); output=output[:, :, i], pointerA=pointerA, pointerB=pointerB)
	end
	for i in axes(f, 1)
		for j in axes(f, 2)
	    	output[i, j, :] = _DT2(output[i, j, :]; output=output[i, j, :], pointerA=pointerA)
		end
	end
	return output
end

# ╔═╡ 5ce9d6c1-0b5a-436a-b035-e8ed9b946200
let
	f = [1, 1, 0, 0, 0, 1, 1]
	arg1, arg2, arg3 = zeros(length(f)), 1, 1
	tfm = Wenbo()
	test = transform(boolean_indicator(f), tfm; output=arg1, pointerA=arg2, pointerB=arg3)
end

# ╔═╡ 25dfa9da-8e5f-453f-bb4a-9158708ebe06
let
	f = [0, 0, 0, 1]
	arg1, arg2, arg3 = zeros(length(f)), 1, 1
	tfm = Wenbo()
	test = transform(boolean_indicator(f), tfm; output=arg1, pointerA=arg2, pointerB=arg3)
end

# ╔═╡ e96f94e9-f719-46d6-94ee-e54fd2ade545
let
	f = [1, 0, 0, 0]
	arg1, arg2, arg3 = zeros(length(f)), 1, 1
	tfm = Wenbo()
	test = transform(boolean_indicator(f), tfm; output=arg1, pointerA=arg2, pointerB=arg3)
end

# ╔═╡ 7a1a556f-fa9e-4cec-88e8-5c74c10e0c4e
let
	f = [1, 1, 0, 0, 0, 1, 1]
	arg1, arg2, arg3 = zeros(length(f)), 1, 1
	tfm = Wenbo()
	@benchmark transform($boolean_indicator($f), $tfm; output=$arg1, pointerA=$arg2, pointerB=$arg3)
end

# ╔═╡ e40f28d3-5f5c-4768-a03b-f08c05b31d31
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
	output, pointerA, pointerB = zeros(size(img)), 1, 1
	tfm = Wenbo()
	test = transform(boolean_indicator(img), tfm; output=output, pointerA=pointerA, pointerB=pointerB)
end

# ╔═╡ ef8cde04-6e7c-4b2a-864b-e3fee501d6a3
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
	output, pointerA, pointerB = zeros(size(img)), 1, 1
	tfm = Wenbo()
	@benchmark transform($boolean_indicator($img), $tfm; output=$output, pointerA=$pointerA, pointerB=$pointerB)
end

# ╔═╡ 32ff4e4c-4207-4a29-9876-1cd51d109449
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
	output, pointerA, pointerB = zeros(size(img)), 1, 1
	tfm = Wenbo()
	test = transform(boolean_indicator(img), tfm; output=output, pointerA=pointerA, pointerB=pointerB)
end

# ╔═╡ 900ef447-b3f9-4ede-820b-ab20002c92b3
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
	output, pointerA, pointerB = zeros(size(img)), 1, 1
	tfm = Wenbo()
	@benchmark transform($boolean_indicator($img), $tfm; output=$output, pointerA=$pointerA, pointerB=$pointerB)
end

# ╔═╡ aeabedd0-5719-46f2-9715-7b21d8e5bf3d
begin
	container=[]
	vol1 = [
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 1 0 0 0 0
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 0 0 0 0 0
]
	vol2 = [
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 1 0 0 0 0
	 0 0 0 1 0 1 0 0 0
	 0 0 0 0 1 0 0 0 0
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 0 0 0 0 0
]
	vol3 = [
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 1 0 0 0 0
	 0 0 0 1 0 1 0 0 0
	 0 0 1 0 0 0 1 0 0
	 0 0 0 1 0 1 0 0 0
	 0 0 0 0 1 0 0 0 0
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 0 0 0 0 0
]
	vol4 = [
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 1 0 0 0 0
	 0 0 0 1 0 1 0 0 0
	 0 0 1 0 0 0 1 0 0
	 0 1 0 0 0 0 0 1 0
	 0 0 1 0 0 0 1 0 0
	 0 0 0 1 0 1 0 0 0
	 0 0 0 0 1 0 0 0 0
	 0 0 0 0 0 0 0 0 0
]
	vol5 = [
	 0 0 0 0 1 0 0 0 0
	 0 0 0 1 0 1 0 0 0
	 0 0 1 0 0 0 1 0 0
	 0 1 0 0 0 0 0 1 0
	 1 0 0 0 0 0 0 0 1
	 0 1 0 0 0 0 0 1 0
	 0 0 1 0 0 0 1 0 0
	 0 0 0 1 0 1 0 0 0
	 0 0 0 0 1 0 0 0 0
]
	#level 1
	push!(container, vol1)
	#level 2
	push!(container, vol2)
	#level 3
	push!(container, vol3)
	#level 4
	push!(container, vol4)
	#level 5
	push!(container, vol5)
	#level 6
	push!(container, vol4)
	#level 7
	push!(container, vol3)
	#level 8
	push!(container, vol2)
	#level 9
	push!(container, vol1)
	test_vol1 = cat(container..., dims=3)
end;

# ╔═╡ ac8ed96d-bf48-4a2e-81f0-ae612024177c
@bind c PlutoUI.Slider(1:size(test_vol1, 3), default=1, show_value=true)

# ╔═╡ fad6db76-5bd3-4350-8180-eb9735f6a749
# heatmap(test_vol1[:, :, c])

# ╔═╡ 33ce5963-4fb7-44e8-9206-d4f6d828db97
# dt_vol_3d = DT3(boolean_indicator(test_vol1))

# ╔═╡ ab59045a-448e-4d09-b779-be372b7b9793
# @bind b PlutoUI.Slider(1:size(dt_vol_3d, 3), default=1, show_value=true)

# ╔═╡ 66c52751-0b74-4204-9eba-71d577c0ecd0
# heatmap(dt_vol_3d[:, :, b])

# ╔═╡ b4d5b33a-4126-46d4-9128-ce778ccbfed8
md"""
### Tests
"""

# ╔═╡ 8e8512c5-e547-4a67-9ec0-936bbe6e58ad
md"""
In 3D, if you stack 2D arrays along with their inverses in a repeating way, the distance transform should result in only zeros and ones. This is because the background element of a 2D slice is always going to be adjacent to a foreground element of the adjacent 2D slice since the adjacent slice is simply the inverted array. This is a good way to test if the 3D form of our distance transform is working or not
"""

# ╔═╡ c3db3faf-4c72-43c9-a0ee-0f65d1b188ef
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
	output, pointerA, pointerB = zeros(size(vol_inv)), 1, 1
	tfm = Wenbo()
	test = transform(boolean_indicator(vol_inv), tfm; output=output, pointerA=pointerA, pointerB=pointerB)
end

# ╔═╡ d5ed3839-9852-4733-ac62-7ad40b6d1541
let
	a1 = [
		 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
		 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
		 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
		 1.0  1.0  1.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0  1.0
		 1.0  1.0  0.0  1.0  1.0  0.0  1.0  1.0  1.0  1.0  1.0
		 1.0  1.0  0.0  1.0  1.0  0.0  0.0  0.0  1.0  1.0  1.0
		 1.0  1.0  0.0  1.0  1.0  1.0  1.0  0.0  1.0  1.0  1.0
		 1.0  1.0  0.0  1.0  1.0  1.0  1.0  0.0  1.0  1.0  1.0
		 1.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0
		 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
		 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
	]
	a2 = [
		 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
		 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
		 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
		 0.0  0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0
		 0.0  0.0  1.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0
		 0.0  0.0  1.0  0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0
		 0.0  0.0  1.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0
		 0.0  0.0  1.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0
		 0.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0
		 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
		 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
	]
	ans = cat(a1, a2, dims=3)
	container_a = []
	for i in 1:10
		push!(container_a, ans)
	end
	answer = cat(container_a..., dims=3)
end

# ╔═╡ 82061388-cab2-4276-a67f-58279a494b4f
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
	output, pointerA, pointerB = zeros(size(vol_inv)), 1, 1
	tfm = Wenbo()
	@benchmark transform($boolean_indicator($vol_inv), $tfm; output=$output, pointerA=$pointerA, pointerB=$pointerB)
end

# ╔═╡ Cell order:
# ╠═aec0d43c-16e0-4092-b81e-6dddbe41d3db
# ╠═45c7f27a-b43c-4781-918a-51aebd273014
# ╠═45b90a66-956a-4b35-9e88-7621fd5c9c1a
# ╠═9614e200-5b49-4747-be88-2b50becc048f
# ╟─e75c9e76-3634-4bde-abb8-5b4827eb089e
# ╠═e5435c7d-5f4d-4698-936f-3ae9906fa9df
# ╠═e5b89de2-64de-42d1-92d8-001a7e7d1bff
# ╟─ceda014b-5b22-4b76-beb0-766a1b138053
# ╠═5ce9d6c1-0b5a-436a-b035-e8ed9b946200
# ╠═25dfa9da-8e5f-453f-bb4a-9158708ebe06
# ╠═e96f94e9-f719-46d6-94ee-e54fd2ade545
# ╠═7a1a556f-fa9e-4cec-88e8-5c74c10e0c4e
# ╟─2d02a6fd-7a23-4790-b8da-7b7150482a85
# ╠═535c4318-d29e-4640-9105-66b16d13bf88
# ╠═b3ba99a1-cbc9-484f-93a5-7a235905cbf8
# ╟─fdc77c42-616a-4822-af8c-1f02bbd8b4a4
# ╠═e40f28d3-5f5c-4768-a03b-f08c05b31d31
# ╠═ef8cde04-6e7c-4b2a-864b-e3fee501d6a3
# ╠═32ff4e4c-4207-4a29-9876-1cd51d109449
# ╠═900ef447-b3f9-4ede-820b-ab20002c92b3
# ╟─71072c67-6584-4fc7-ad4d-b3d362333562
# ╠═f5ff25be-94d2-4e97-bb4d-df36abb4c0a8
# ╟─aeabedd0-5719-46f2-9715-7b21d8e5bf3d
# ╟─ac8ed96d-bf48-4a2e-81f0-ae612024177c
# ╠═fad6db76-5bd3-4350-8180-eb9735f6a749
# ╠═33ce5963-4fb7-44e8-9206-d4f6d828db97
# ╠═ab59045a-448e-4d09-b779-be372b7b9793
# ╠═66c52751-0b74-4204-9eba-71d577c0ecd0
# ╟─b4d5b33a-4126-46d4-9128-ce778ccbfed8
# ╟─8e8512c5-e547-4a67-9ec0-936bbe6e58ad
# ╠═c3db3faf-4c72-43c9-a0ee-0f65d1b188ef
# ╠═d5ed3839-9852-4733-ac62-7ad40b6d1541
# ╠═82061388-cab2-4276-a67f-58279a494b4f
