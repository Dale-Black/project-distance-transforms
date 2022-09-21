### A Pluto.jl notebook ###
# v0.19.10

using Markdown
using InteractiveUtils

# ╔═╡ aec0d43c-16e0-4092-b81e-6dddbe41d3db
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate("/Users/wenboli/Desktop/ssd/dependency")
	# Pkg.instantiate()
	Pkg.Registry.update()
	# Pkg.add("PlutoUI")
	# Pkg.add("CairoMakie")
	# Pkg.add("Luxor")
	# Pkg.add("BenchmarkTools")
	# Pkg.add("ColorTypes")
	# Pkg.add(url="https://github.com/Dale-Black/ActiveContours.jl")
end

# ╔═╡ b05b4228-fc30-4a74-8a99-f9b8d53aff83
begin
	using PlutoUI
	using CairoMakie
	using LinearAlgebra
	using Luxor
	using BenchmarkTools
	using ColorTypes
	using ActiveContours
	using Printf
end

# ╔═╡ 45c7f27a-b43c-4781-918a-51aebd273014
TableOfContents()

# ╔═╡ 1b7fee3a-1b11-411c-97a9-e6ede29ee56e
# ╠═╡ disabled = true
#=╠═╡
begin
	customINF = Float64(99999.0)
	customNegINF = Float64(-99998.0)
end
  ╠═╡ =#

# ╔═╡ 45b90a66-956a-4b35-9e88-7621fd5c9c1a
boolean_indicator(f) = @. ifelse(f == 0, 1f10, 0f0)

# ╔═╡ f6bc658b-dfd5-44e3-a6ff-26797729ff81
# boolean_indicator_Ver1(f) = @. ifelse(f == 0, Float64(-1), Float64(0))
boolean_indicator_Ver1(f) = @. ifelse(f == 0, -1f0, 0f0)

# ╔═╡ a6c9dc03-74ce-4174-8fd9-6895d117aa4a
# ╠═╡ disabled = true
#=╠═╡
function boolean_indicator_Ver2(f)
	for i = 1 : length(f)
		if (f[i] == 0)
			f[i] = 1f10
		else
			f[i] = 0f0
		end
	end
	return f
end
  ╠═╡ =#

# ╔═╡ d051233a-8363-4dd8-807f-2d273e55e29f
# ╠═╡ disabled = true
#=╠═╡
function boolean_indicator_Ver3(f)
	# this is for 1D only
	g = zeros(Float64, length(f))
	# f = Float64.(f)
	for i = 1 : length(f)
		if (f[i] == 0)
			g[i] = -1
		# else
		# 	f[i] = 0
		end
	end
	return g
end
  ╠═╡ =#

# ╔═╡ 6e2adb2c-93f4-4a19-b93c-75b75a9116f3
@benchmark boolean_indicator([1 1 0 0 0 1 0 0 1 1 1 0 1 1])

# ╔═╡ f6631dd6-cb06-4c84-a5db-38ad6af64b68
@benchmark boolean_indicator_Ver1([1 1 0 0 0 1 0 0 1 1 1 0 1 1])

# ╔═╡ f873e7c2-7dac-4925-b6b5-5887b461418b
@benchmark boolean_indicator_Ver2([1 1 0 0 0 1 0 0 1 1 1 0 1 1])

# ╔═╡ 0c569423-c87f-4139-a2e6-2e1a897d5f39
@benchmark boolean_indicator_Ver3([1 1 0 0 0 1 0 0 1 1 1 0 1 1])

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
function DT1(f; D=zeros(length(f)), v=ones(Int32, length(f)), z=ones(length(f)+1))
	z[1] = -1f10
	z[2] = 1f10
	k = 1; # Index of the rightmost parabola in the lower envelope
	for q = 2:length(f)
		s = ((f[q] + q^2) - (f[v[k]] + v[k]^2)) / (2*q - 2*v[k])
	    while s ≤ z[k]
	        k -= 1
	        s = ((f[q] + q^2) - (f[v[k]] + v[k]^2)) / (2*q - 2*v[k])
	    end
	    k += 1
		#buffer here
	    v[k] = q
	    z[k] = s
		z[k+1] = 1f10
	end
	k = 1
	for q in 1:length(f)
	    while z[k+1] < q
	        k = k+1
	    end
		
	    D[q] = (q-v[k])^2 + f[v[k]]
		#v[k] >= k is true all the time
	end
	return D
end

# ╔═╡ 098b8119-26e8-4002-84c4-c8478d3c5019
begin
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
	function DT1Wenbo(f; output=zeros(length(f)), pointerA=1, pointerB=1)
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
end

# ╔═╡ 93de6d70-46ff-414f-9185-72975b252cbe
begin
	function DT1helper!(f, i, j)
		if (i==-1 && j==-1)
			i=1
			while(i<=length(f))
				f[i]=1f10
				i+=1
			end
		elseif(i==-1)
			temp=1
			while(j>0)
				f[j]=temp^2
				j-=1
				temp+=1
			end
		elseif(j==-1)
			temp=1
			while(i<=length(f))
				f[i]=temp^2
				i+=1
				temp+=1
			end
		else
			temp=1
			while(i<=j)
				f[i]=f[j]=temp^2
				temp+=1
				i+=1
				j-=1
			end
		end
	end
	function DT1Wenbo!(f)
		#assume length(f)>0
		#This is a one pass algorithm
		#time complexity=O(n), Space complexity=O(1)
		pointerA = pointerB = 1
		while (pointerA<=length(f))
			if(f[pointerA] == 0)
				pointerA+=1
				pointerB+=1
			else
				while(pointerB <= length(f) && f[pointerB]==1f10)
					pointerB+=1
				end
				if (pointerB > length(f))
					if (pointerA == 1)
						DT1helper!(f, -1, -1)
					else
						DT1helper!(f, pointerA, -1)
					end
				else
					if (pointerA == 1)
						DT1helper!(f, -1, pointerB-1)
					else
						DT1helper!(f, pointerA, pointerB-1)
					end
				end
				pointerA=pointerB
			end
		end
	end
end

# ╔═╡ 5dd214c2-91ee-43cf-9e1f-97dd64fbd383
begin
	function DT1helper_Ver2!(f, i, j)
		if (i==-1 && j==-1)
			i=1
			while(i<=length(f))
				f[i]=-1.0
				i+=1
			end
		elseif(i==-1)
			temp=1
			while(j>0)
				f[j]=temp^2
				j-=1
				temp+=1
			end
		elseif(j==-1)
			temp=1
			while(i<=length(f))
				f[i]=temp^2
				i+=1
				temp+=1
			end
		else
			temp=1
			while(i<=j)
				f[i]=f[j]=temp^2
				temp+=1
				i+=1
				j-=1
			end
		end
	end
	function DT1Wenbo_Ver2!(f)
		#assume length(f)>0
		#This is a one pass algorithm
		#time complexity=O(n), Space complexity=O(1)
		pointerA = pointerB = 1
		while (pointerA<=length(f))
			if(f[pointerA] == 0)
				pointerA+=1
				pointerB+=1
			else
				while(pointerB <= length(f) && f[pointerB]<0)
					pointerB+=1
				end
				if (pointerB > length(f))
					if (pointerA == 1)
						DT1helper_Ver2!(f, -1, -1)
					else
						DT1helper_Ver2!(f, pointerA, -1)
					end
				else
					if (pointerA == 1)
						DT1helper_Ver2!(f, -1, pointerB-1)
					else
						DT1helper_Ver2!(f, pointerA, pointerB-1)
					end
				end
				pointerA=pointerB
			end
		end
	end
end

# ╔═╡ 79041c45-d062-4b0d-8643-aca2a6b6ccf1
#=╠═╡
begin
	function encodeF(input)
		input = min(input, customINF)
		idx = Int8(0)
		while(input>=1)
			input /= 10
			idx += 1 
		end
		input /= 10
		input += idx
		return input/10
	end
	
	function decodeF_2(input)
		input = abs(input)
		input *= 10
		idxF = trunc(Int8, input % 10)
		input *= 10
		idxV = trunc(Int8, input % 10)
		
		input -= trunc(input)
		while (idxF > 0)
			idxF -= 1
			input *= 10
		end
		if (idxV == 0)
			return Int(round(input))
		end
		return Int(floor(input))
	end
end
  ╠═╡ =#

# ╔═╡ 230a3b21-01d3-4728-a07b-8382aa1e5db1
begin 
	function encodeV(input, v=1)
		sign = 1.0
		if (input < 0)
			sign = -1
			input = -input
		end
		new_idxV = Int8(0)
		while(v>=1)
			v /= 10
			new_idxV += 1 
		end
		input *= 10
		idxF = trunc(Int8, input % 10)
		input *= 10
		idxV = trunc(Int8, input % 10)
		
		input += new_idxV
		input -= idxV
		if (idxV == 0)
			input = round(input, digits=Int(idxF))
		else
			input = floor(input, digits=Int(idxF))
		end
		while (idxF > 0)
			idxF -= 1
			v /= 10
		end
		input+=v
		return input/100 * sign
	end
	function decodeV_2(input)
		input = abs(input)
		input *= 10
		idxF = trunc(Int8, input % 10)
		input *= 10
		idxV = trunc(Int8, input % 10)
		
		input -= trunc(Int, input)
		while (idxF > 0)
			idxF -= 1
			input *= 10
		end
		
		input -= trunc(Int, input)
		while (idxV > 0)
			idxV -= 1
			input *= 10
		end
		return Int16(round(input))
	end
end

# ╔═╡ debda360-c879-41cc-a8ed-281b75b33934
function encodeFV(input)
		input = input * 10 + 1
		idx = Int8(-1)
		while(input>=1)
			input /= 10
			idx += 1 
		end
		input += 1
		input /= 10
		input += idx
		return input/10
	end

# ╔═╡ fcea1ad8-3fb1-4e82-b37f-80bed8a11df2
begin
	function encodeZ(input, z=1)
		input -= trunc(input)
		if (z>0)
			return abs(input) + z
		end
		return -abs(input) + z
	end
end

# ╔═╡ c3ab6a19-873f-41c1-b728-caf89c52bf25
# ╠═╡ disabled = true
#=╠═╡
function DT1ver3_copy(f)
	ct=1
	D=zeros(length(f))

	#z[1] = -Inf32
	f[1] = encodeFV(f[1])
	f[1] += 9999.0
	f[1] = -f[1]
	
	#z[2] = Inf32
	f[2] = encodeFV(f[2])
	f[2] += 9999.0

	#v=ones(Int32, length(f)), z=ones(length(f)+1)
	for i = 3:length(f)
		f[i] = encodeFV(f[i])
		f[i] += 1.0
	end
	lastZ=1
	
	k = 1; # Index of the rightmost parabola in the lower envelope
	for q = 2:length(f)

		#s = ((f[q] + q^2) - (f[v[k]] + v[k]^2)) / (2*q - 2*v[k])
		fq = decodeF_2(f[q])

		
		print("data[")
		print(k)
		print("]=")
		print(f[k])
		print("\n")
		
		print("v[")
		print(k)
		print("]=")
		print(decodeV_2(f[k]))
		print("\n")
		
		vk = decodeV_2(f[k])
		comp1 = min(9999, muladd(q, q, fq))
		comp2 = min(9999, muladd(vk, vk, decodeF_2(f[vk])))
		comp3 = 2*q - 2*vk
		s = (comp1 - comp2) / comp3
		s = max(-9998.0, s)
		s = min(9998.0, s)
		currZ = lastZ
		if k ≤ length(f)
			currZ = trunc(f[k])
		end
	
		print("\n\t")
		print(fq)#0
		print("\n\t")
		print(q)#6
		print("\n\t")
		print(decodeF_2(f[vk]))#0
		print("\n\t")
		print(vk)#2
		print("\n")
		
		print("q = ")
		print(q)
		print(", k = ")
		print(k)
		print("\n")
		
		print(ct)
		ct+=1
		print("*****s = ")
		print(s)
		print(", z[")
		print(k)
		print("] = ")
		print(currZ)
		print("*****\n")
		print("*****comp1 = ")
		print(comp1)
		print(", comp2 = ")
		print(comp2)
		print(", comp3 = ")
		print(comp3)
		print("*****\n")
		
	    while s ≤ currZ
	        k -= 1
			vk = decodeV_2(f[k])
		comp2 = min(9999, muladd(vk, vk, decodeF_2(f[vk])))
		comp3 = 2*q - 2*vk
	        s = (comp1 - comp2) / comp3
			s = max(-9998, s)
			s = min(9998.0, s)
			
			print("q = ")
			print(q)
			print(", k = ")
			print(k)
			print("\n")
			
			currZ = lastZ
			if k ≤ length(f)
				currZ = trunc(f[k])
			end
				
			print("\n\t")
		print(fq)
		print("\n\t")
		print(q)
		print("\n\t")
		print(decodeF_2(f[vk]))
		print("\n\t")
		print(vk)
		print("\n")

		print(ct)
		ct+=1
		print("*****s = ")
		print(s)
		print(", z[")
		print(k)
		print("] = ")
		print(currZ)
		print("*****\n")
		print("*****comp1 = ")
		print(comp1)
		print(", comp2 = ")
		print(comp2)
		print(", comp3 = ")
		print(comp3)
		print("*****\n")
	    end
	    k += 1
	    #v[k] = q
		f[k] = encodeV(f[k], q)
		print("v[")
		print(k)
		print("]=")
		print(q)
		print("\n")
		
	    #z[k] = s
		print("\tdata[")
		print(k)
		print("]=")
		print(f[k])
		print("\n")
		
		f[k] = encodeZ(f[k],trunc(s))
		print("\tz[")
		print(k)
		print("]=")
		print(s)
		print("\n")

		print("\tdata[")
		print(k)
		print("]=")
		print(f[k])
		print("\n")
		
		#z[k+1] = Inf32
		if k+1 ≤ length(f)
			f[k+1] = encodeZ(f[k+1],9999)
		else
			lastZ = 9999
		end
		print("z[")
		print(k+1)
		print("]=9999\n\n")
	end

	for i = 1 : 14
		print("f[")
		print(i)
		print("]=")
		print(decodeF_2(f[i]))
		print("\n")
	end
	print("\n")
	for i = 1 : 14
		print("v[")
		print(i)
		print("]=")
		print(decodeV_2(f[i]))
		print("\n")
	end
	print("\n")
	for i = 1 : 14
		print("z[")
		print(i)
		print("]=")
		print(trunc(f[i]))
		print("\n")
	end
	print(lastZ)
	print("\n\n")

	
	k = 1
	for q in 1:length(f)
	    # while z[k+1] < q
		
		currZ = lastZ
		if k+1 ≤ length(f)
			currZ = trunc(f[k+1])
		end
		
	    while currZ < q
	        k = k+1
			currZ = lastZ
			if k+1 ≤ length(f)
				currZ = trunc(f[k+1])
			end
			#buffer here
	    end
	    #D[q] = (q-v[k])^2 + f[v[k]] #v[k] >= k is true all the time
		vk = decodeV_2(f[k])
		comp1 = q-vk
		D[q] = muladd(comp1, comp1, decodeF_2(f[vk]))
	end
	return D
end
  ╠═╡ =#

# ╔═╡ 722a98fa-7c00-46ad-b0d6-ba83d258204c
# ╠═╡ disabled = true
#=╠═╡
function DT1ver3(f)
	D=zeros(length(f))

	#z[1] = -Inf32
	f[1] = encodeFV(f[1])
	f[1] += 9999.0
	f[1] = -f[1]
	
	#z[2] = Inf32
	f[2] = encodeFV(f[2])
	f[2] += 9999.0

	#v=ones(Int32, length(f)), z=ones(length(f)+1)
	for i = 3:length(f)
		f[i] = encodeFV(f[i])
		f[i] += 1.0
	end
	lastZ=1
	
	k = 1; # Index of the rightmost parabola in the lower envelope
	for q = 2:length(f)

		#s = ((f[q] + q^2) - (f[v[k]] + v[k]^2)) / (2*q - 2*v[k])
		fq = decodeF_2(f[q])
		
		vk = decodeV_2(f[k])
		comp1 = min(9999, muladd(q, q, fq))
		comp2 = min(9999, muladd(vk, vk, decodeF_2(f[vk])))
		comp3 = 2*q - 2*vk
		s = (comp1 - comp2) / comp3
		s = max(-9998.0, s)
		s = min(9998.0, s)
		currZ = lastZ
		if k ≤ length(f)
			currZ = trunc(f[k])
		end
		
	    while s ≤ currZ
	        k -= 1
			vk = decodeV_2(f[k])
			comp2 = min(9999, muladd(vk, vk, decodeF_2(f[vk])))
			comp3 = 2*q - 2*vk
	        s = (comp1 - comp2) / comp3
			s = max(-9998, s)
			s = min(9998.0, s)
			
			currZ = lastZ
			if k ≤ length(f)
				currZ = trunc(f[k])
			end
	    end
	    k += 1
		
	    #v[k] = q
		f[k] = encodeV(f[k], q)
		
	    #z[k] = s
		f[k] = encodeZ(f[k],trunc(s))
		
		#z[k+1] = Inf32
		if k+1 ≤ length(f)
			f[k+1] = encodeZ(f[k+1],9999)
		else
			lastZ = 9999
		end
	end
	
	k = 1
	for q in 1:length(f)
	    # while z[k+1] < q
		
		currZ = lastZ
		if k+1 ≤ length(f)
			currZ = trunc(f[k+1])
		end
		
	    while currZ < q
	        k = k+1
			currZ = lastZ
			if k+1 ≤ length(f)
				currZ = trunc(f[k+1])
			end
			#buffer here
	    end
	    #D[q] = (q-v[k])^2 + f[v[k]] #v[k] >= k is true all the time
		vk = decodeV_2(f[k])
		comp1 = q-vk
		D[q] = muladd(comp1, comp1, decodeF_2(f[vk]))
	end
	return D
end
  ╠═╡ =#

# ╔═╡ 37124f17-dee2-40b0-bbb1-b5946c7efa48
begin 
	function DT1Ver4!(f)
		#foward
		temp = 0
		for i = 2:length(f)
			curr = f[i-1]+temp*2+1
			if (curr < f[i])
				f[i] = curr
				temp +=1
			else
				temp = 0
			end
		end
		#backward
		temp = 0
		for i = length(f)-1 : -1 : 1
			curr = f[i+1]+temp*2+1
			if (curr < f[i])
				f[i] = curr
				temp +=1
			else
				temp = 0
			end
		end
	end
end

# ╔═╡ 162718f4-40c7-4bc8-9f8c-f9f4ae56573d
#=╠═╡
let
	# print(encodeVF(99999))
	temp = [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1]
	temp = boolean_indicator_Ver3(temp)
	# print("\ncorrect: \n")
	# print(DT1(temp))
	# print("\n\n")
	DT1Wenbo_Ver2!(temp)
	temp
end
  ╠═╡ =#

# ╔═╡ 2280361b-068d-4d5e-ae30-7e66c42ee665
#=╠═╡
begin
	function testDT1Wenbo!(size,numCases)
		for i in 1:numCases
			testInput=boolean_indicator(rand([0, 1], size))
			rslt = DT1(testInput)
			DT1Wenbo!(testInput)
			if (testInput != rslt)
				println(testInput)
				println(rslt)
				println()
				return "Failed."
			end 
		end
		return "DT1Wenbo! Passed."
	end
	function testDT1Ver4(size,numCases)
		for i in 1:numCases
			testInput=boolean_indicator(rand([0, 1], size))
			rslt = DT1(testInput)
			DT1Ver4!(testInput)
			for j in 1:size
				if (testInput[i] - rslt[i] != 0.0)
					println(testInput)
					println(rslt)
					return "Failed."
				end
			end 
		end
		return "DT1Ver4 Passed."
	end
	function testDT1Wenbo_Ver2!(size,numCases)
		for i in 1:numCases
			testInput = rand([0, 1], size)
			rslt = DT1(boolean_indicator(testInput))
			testInput = boolean_indicator_Ver3(testInput)
			DT1Wenbo_Ver2!(testInput)
			if (testInput != rslt)
				println(testInput)
				println(rslt)
				println()
				return "Failed."
			end 
		end
		return "DT1Wenbo_Ver2! Passed."
	end
end
  ╠═╡ =#

# ╔═╡ 03d7270c-58ee-4ce2-a67d-262bc62c9478
#=╠═╡
testDT1Wenbo!(1024,100)
  ╠═╡ =#

# ╔═╡ c8e739c0-940d-47b4-be2e-b0345a7e362f
#=╠═╡
testDT1Ver4(1024,100)
  ╠═╡ =#

# ╔═╡ ea2e8c7e-23fe-4277-bb4d-ae56123163ab
#=╠═╡
testDT1Wenbo_Ver2!(1024,100)
  ╠═╡ =#

# ╔═╡ 0469116a-7e10-44c5-801d-f5ce683dc486
begin
	f = [1 1 0 0 0 1 0 0 1 1 1 0 1 1];
	D, v, z =zeros(length(f)), ones(Int32, length(f)), ones(length(f));
	arg1 = zeros(length(f));
end

# ╔═╡ 0cca3714-b73a-4f6e-b953-e01d392bdc91
# boolean_indicator(f)
DT1(boolean_indicator(f))

# ╔═╡ 30ae8059-4b4e-4026-966e-7d6320912b5c
@benchmark DT1($boolean_indicator($f); D=$D, v=$v, z=$z) 

# ╔═╡ 7a1a556f-fa9e-4cec-88e8-5c74c10e0c4e
@benchmark DT1Wenbo($boolean_indicator($f); output=$arg1, pointerA=1, pointerB=1) 

# ╔═╡ 5f9287b3-c7c3-4c0d-bf3b-d3161c4bb249
@benchmark DT1Wenbo!($boolean_indicator($f))

# ╔═╡ cc775fe3-563d-46ff-8adc-4164dd0e9ea2
@benchmark DT1Wenbo_Ver2!($boolean_indicator_Ver1($f))

# ╔═╡ c0a2399a-147a-41ca-8a27-4bc6a70dca52
@benchmark DT1Ver4!($boolean_indicator($f)) 

# ╔═╡ 2d02a6fd-7a23-4790-b8da-7b7150482a85
md"""
## 2D
"""

# ╔═╡ 29cb3c08-a702-4564-ad41-664f312913b0
begin
	img2d1 = [
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
	];
	img2d2 = [
		 0 1 1 1 0 0 0 1 1
		 1 1 1 1 1 0 0 0 1
		 1 0 0 0 1 0 0 1 1
		 1 0 0 0 1 0 1 1 0
		 1 0 0 0 1 1 0 1 0
		 1 1 1 1 1 0 0 1 0
		 0 1 1 1 0 0 0 0 1
	];
	img2d3 = [
		0	0	1	0	0	1	0	
		1	0	0	1	1	0	1	
		0	0	0	1	0	1	0	
		0	0	0	0	0	0	0	
		1	0	1	0	1	1	1	
		0	0	1	1	1	1	0	
		0	0	0	1	1	0	0	
	];
end

# ╔═╡ 2d7da285-7078-4567-8cdd-a8cefd737f57
function DT2(img; D=zeros(size(img)))
	for i = 1:size(img, 1)
	    D[i, :] = DT1(img[i, :])
	end
	for j = 1:size(img, 2)
	    D[:, j] = DT1(D[:, j])
	end
	return D
end

# ╔═╡ 535c4318-d29e-4640-9105-66b16d13bf88
begin
	# june 20 new implementation
	function DT2Helper(f, output)
		pointerA = 1;
		while (pointerA<=length(f))
			output[pointerA]=f[pointerA]
			#it is needless to deal with distance <= 1
			if(f[pointerA] > 1)
				temp = 1
				if (2 * pointerA > length(f))
					#2nd half
					while (output[pointerA]>1 && temp <= length(f) - pointerA)
						#right
						if (f[pointerA+temp]<output[pointerA])
							output[pointerA]=min(output[pointerA], f[pointerA+temp]+temp^2)
						end
						#left
						if (f[pointerA-temp]<output[pointerA])
							output[pointerA]=min(output[pointerA], f[pointerA-temp]+temp^2)
						end
							temp += 1
					end
					#continue searching for left
					if(output[pointerA]>1)
						while (output[pointerA]>1 && temp < pointerA)
							if (f[pointerA-temp]<output[pointerA])
								output[pointerA]=min(output[pointerA], f[pointerA-temp]+temp^2)
							end
							temp += 1
						end
					end
				else
					#1st half
					while (output[pointerA]>1 && temp < pointerA)
						#right
						if (f[pointerA+temp]<output[pointerA])
							output[pointerA]=min(output[pointerA], f[pointerA+temp]+temp^2)
						end
						#left
						if (f[pointerA-temp]<output[pointerA])
							output[pointerA]=min(output[pointerA], f[pointerA-temp]+temp^2)
						end
						temp += 1
					end
					#continue searching for right
					if(output[pointerA]>1)
						while (output[pointerA]>1 && temp <= length(f) - pointerA)
							if (f[pointerA+temp]<output[pointerA])
								output[pointerA]=min(output[pointerA], f[pointerA+temp]+temp^2)
							end
							temp += 1
						end
					end
				end
			end
			pointerA+=1
		end
		return output
	end
	function DT2Wenbo(img; D=zeros(size(img)), pointerA=1, pointerB=1)
		# This is a worst case = O(n^3) implementation
		for i = 1:size(img, 1)
		    D[i, :] = DT1Wenbo(img[i, :]; output=D[i, :], pointerA=pointerA, pointerB=pointerB) 
		end
		# june 20 new implementation
		for j = 1:size(img, 2)
		    D[:, j] = DT2Helper(D[:, j], D[:, j]) 
		end
		return D
	end
end

# ╔═╡ 9eef234c-f269-4437-9e0e-299447cd1d10
begin
	# original     f        -1      0 or 1
	#
	#              |        |         no need to encode
	#              v        v         
	#
	# encode to   f.dD      -0.dD     N/A 
	#
	#              |        |         no need to decode
	#              v        v         
	#
	# decodef to    f       -1      N/A 
	# decodeD to    D       D       N/A 
	function encodeVer2(leftf, rightD)
		idx = 0
		while(rightD>=1)
			rightD/=10
			idx+=1 
		end
		if leftf <0
			return -idx/10-rightD/10
		end
		return leftf+idx/10+rightD/10
	end
	function decodeDVer2(curr)
		curr = abs(curr)
		curr *= 10
		temp = Int(floor(curr))
		curr -= temp
		for i = 1 : temp%10
			curr*=10
		end
		return round(curr)
	end
	function decodefVer2(curr)
		if curr < 0
			return -1
		end
		return trunc(curr)
	end
	function DT2HelperVer2!(f)
		#foward
		p = 0
		for i = 2:length(f)
			newD = f[i]
			if (trunc(newD*10%10))!= 0
				newD = decodeDVer2(f[i])
			end
			offset = 0
			if 0<= newD <= 1
				p=0
				continue
			end
			for j = 1+p : -1 : 1
				offset += 1
				oldf = f[i-offset]
				if (trunc(oldf*10%10))!= 0
					oldf = decodefVer2(oldf)
				end
				if oldf < 0
					continue
				end
				currD = oldf+offset^2
				# @printf "f[%.0f] = %.2f\t%.2f+%.2f^2=%.2f\n" i newD oldf offset currD
				if newD==-1.0 || currD < newD
					newD = currD
					p = offset
				end
				if newD <= 1.0
					break
				end
			end
			if (offset == 0)
				p = 0
			else					
				# @printf "Encoding leftf = %.2f, rightD = %.2f ----> %.10f\n" f[i] newD encodeVer2(f[i], newD)
				f[i] = encodeVer2(f[i], newD)
			end
		end
		#backward
		p = 0
		for i = length(f)-1 : -1 : 1
			newD = f[i]
			if (trunc(newD*10%10))!= 0
				newD = decodeDVer2(f[i])
			end
			offset = 0
			if 0<= newD <= 1
				p=0
				continue
			end
			for j = 1+p : -1 : 1
				offset += 1
				oldf = f[i+offset]
				if (trunc(oldf*10%10))!= 0
					oldf = decodefVer2(oldf)
				end
				if oldf < 0
					continue
				end
				currD = oldf+offset^2
				# @printf "f[%.0f] = %.2f\t%.2f+%.2f^2=%.2f\n" i newD oldf offset currD
				if newD==-1.0 || currD < newD
					newD = currD
					p = offset
				end
				if newD <= 1.0
					break
				end
			end
			if (offset == 0)
				p = 0
			else					
				oldf = f[i]
				if (trunc(oldf*10%10))!= 0
					oldf = decodefVer2(oldf)
				end
				# @printf "Encoding leftf = %.2f, rightD = %.2f ----> %.10f\n" oldf newD encodeVer2(oldf, newD)
				f[i] = encodeVer2(oldf, newD)
			end
		end
		# decode to return
		for i = 1 : length(f)
			if (trunc(f[i]*10%10))!= 0
				f[i] = decodeDVer2(f[i])
			elseif f[i]  == -1.0
				f[i] = 1f10
			end
		end
	end

	function DT2WenboVer2!(f2d)
		for i = 1:size(f2d, 1)
			DT1Wenbo_Ver2!(@view(f2d[i, :]))
		end
		for i = 1:size(f2d, 2)
			DT2HelperVer2!(@view(f2d[:, i])) 
		end
	end
end

# ╔═╡ bce0d88f-4384-4b22-8b19-a3980e81212b
# ╠═╡ disabled = true
#=╠═╡
begin
	function whileloop()
		i = 0
		ct = 0
		while (i < 7)
			i += 1
			ct += i
		end
	end
	function forloop()
		ct = 0
		for i = 1: 7
			ct += i
		end
	end
end
  ╠═╡ =#

# ╔═╡ 40c9bdbd-6898-4919-9261-671ff7702296
@benchmark whileloop()

# ╔═╡ 416e2c45-2f56-4136-bfe5-e411784e9829
@benchmark forloop()

# ╔═╡ ec92e0e4-95f9-4dfd-b62a-2387493c426a
# ╠═╡ disabled = true
#=╠═╡
@benchmark encodeVer3(2,15)
  ╠═╡ =#

# ╔═╡ 24146d99-44f0-486d-b8ab-5d0f1c54e26b
# ╠═╡ disabled = true
#=╠═╡
@benchmark encode(2,15)
  ╠═╡ =#

# ╔═╡ 7f835715-f871-4abd-b8fa-d383a9a52b3c
# ╠═╡ disabled = true
#=╠═╡
@benchmark decodeVer3(-2.2150000000000003)
  ╠═╡ =#

# ╔═╡ 3eb49acc-ae0f-4ba4-aa9e-a8899d8d4960
# ╠═╡ disabled = true
#=╠═╡
@benchmark decodeVer3(-2.2150000000000003)
  ╠═╡ =#

# ╔═╡ 6c041277-0246-4e22-b296-2a4a8160d247
begin
	function encodeVer3(leftD, rightf)
		if rightf == 1f10
			return -leftD
		end
		idx = 0
		while(rightf>1)
			rightf/=10
			idx+=1 
		end
		return -leftD-idx/10-rightf/10
	end
	function decodeVer3(curr)		# -10.0 			-2.2150000000000003
		curr *= -10   				# 100    			22.150000000000003
		temp = Int(floor(curr))		# 100    			22
		curr -= temp 				#   0                0.15
		if curr == 0
			return 1f10
		end
		temp %= 10
		while temp > 0
			temp -= 1
			curr*=10
		end
		return round(curr)
	end
	# Sep 16 modified
	
	# original     f        1f10       0 or 1
	#
	#              |        |       no need to encode
	#              v        v         
	#
	# encode to   -D.fF    -D.0       0 or 1
	#
	#              |        |      no need to decode
	#              v        v         
	#
	# decode to    f       1f10       0 or 1
	
	function DT2HelperVer3!(f)
		#foward
		pointerA = 2
		l = length(f)
		while pointerA <= l && f[pointerA] <= 1
			pointerA += 1
		end
		p = 0
		while pointerA <= l
			newDistance = f[pointerA]
			offset = p+1
			# changed = false
			while offset > 0 && newDistance > 1
				oldf = f[pointerA - offset]
				if oldf < 0
					oldf = decodeVer3(oldf)
				end
				if oldf < 1f10
					temp = muladd(offset, offset, oldf)
					if temp < newDistance
						newDistance = temp
						p = offset
						# changed = true
					end
				end
				offset -= 1
			end
			#enocde newDistance to f[pointerA]
			f[pointerA] = encodeVer3(newDistance, f[pointerA])
			pointerA += 1;	
			while pointerA <= l && f[pointerA] <= 1
				pointerA += 1
				p = 0
			end
		end
		#backward
		pointerA = l-1
		while pointerA > 0 && floor(abs(f[pointerA])) <= 1
			pointerA -= 1
		end
		p=0
		while pointerA > 0
			newDistance = floor(abs(f[pointerA]))
			currDistance = newDistance
			offset = p+1
			# changed = false
			while offset > 0 && newDistance > 1
				oldf = f[pointerA + offset]
				if oldf < 0
					oldf = decodeVer3(oldf)
				end
				if oldf <1f10
					temp = muladd(offset, offset, oldf)
					if temp < newDistance
						newDistance = temp
						p = offset
						# changed = true
					end
				end
				offset -= 1
			end
			#enocde newDistance to f[pointerA]
			f[pointerA] = encodeVer3(newDistance, currDistance)
			pointerA -= 1;
			while pointerA > 0 && floor(abs(f[pointerA])) <= 1
				pointerA -= 1
				p = 0
			end
		end
		# decode to return
		i = 0
		while i < l
			i += 1
			f[i] = floor(abs(f[i]))
		end
	end
	
	function DT2WenboVer3!(f2d)
		for i = 1:size(f2d, 1)
			DT1Wenbo!(@view(f2d[i, :]))
		end
		for i = 1:size(f2d, 2)
			DT2HelperVer3!(@view(f2d[:, i])) 
		end
	end
end

# ╔═╡ 56a72a97-1505-4927-9106-9e4e0111dba9
begin
	# Sep 16 modified
	
	# original     f        1f10       0 or 1
	#
	#              |        |       no need to encode
	#              v        v         
	#
	# encode to   -D.fF    -D.0       0 or 1
	#
	#              |        |      no need to decode
	#              v        v         
	#
	# decode to    f       1f10       0 or 1

	
	function encode(curr, prev)
		idx = 0
		while(prev>1)
			prev/=10
			idx+=1 
		end
		return -curr-idx/10-prev/10
	end
	function decode(curr)
		curr *= -10
		temp = Int(floor(curr))
		curr -= temp
		temp %= 10
		while temp > 0
			temp -= 1
			curr*=10
		end
		return round(curr)
	end
	# july 27 modified
	function DT2Helper!(f)
		l = length(f) # faster?*************
		pointerA = 1
		while pointerA<=l && f[pointerA] <= 1
			pointerA += 1
		end
		while pointerA<=l
			curr = f[pointerA]
			prev = curr
			#left bound : temp < pointerA
			#right bound : temp <= l - pointerA
			#left
			temp = 1
				while (temp < pointerA && muladd(temp,temp,-curr)<0)
					#left
					fi=f[pointerA-temp]
					prevprev = fi<0 ? decodeVer3(fi) : fi
					curr = min(curr, prevprev+temp*temp)
					temp += 1
				end
				#right
				temp = 1
				while (temp <= l - pointerA && muladd(temp,temp,-curr)<0)
					curr = min(curr, f[pointerA+temp]+temp*temp)
					temp += 1
				end
			# if (prev != curr && prev >1)
			f[pointerA] = encodeVer3(curr, prev)
			# end
			pointerA+=1
			while pointerA<=l && f[pointerA] <= 1
				pointerA += 1
			end
		end
		i = 0
		while i<l
			i+=1
			f[i] = floor(abs(f[i]))
		end
	end
	function DT2Wenbo!(f2d)
		i = 0
		s1 = size(f2d, 1)
		while i < s1
			i += 1
			DT1Wenbo!(@view(f2d[i, :]))
		end
		j = 0
		s2 = size(f2d, 2)
		while j < s2
			j += 1
			DT2Helper!(@view(f2d[:, j])) 
		end
	end
end

# ╔═╡ 0509a408-2dbc-483f-bd5f-b9832268845a
function DT2HelperVer4!(f)
		l = length(f) # faster?*************
		pointerA = 1
		while pointerA<=l && f[pointerA] <= 1
			pointerA += 1
		end
		p = 0
		while pointerA<=l
			curr = f[pointerA]
			prev = curr
			#left bound : temp < pointerA
			#right bound : temp <= l - pointerA
			#left
			temp = min(pointerA-1, p+1)
			p = 0
			# while (temp < dup_P+1 && curr>temp*temp)
			while (0 < temp)
				fi = f[pointerA-temp]
				fi = fi < 0 ? decodeVer3(fi) : fi
				newDistance = muladd(temp, temp, fi)
				if newDistance < curr
					curr = newDistance
					p = temp
				end
				temp -= 1
			end
			#right
			temp = 1
			templ = l - pointerA
			while (temp <= templ && muladd(temp, temp, -curr) < 0)
				curr = min(curr, muladd(temp, temp, f[pointerA+temp]))
				temp += 1
			end
			# if (prev != curr && prev >1)
			f[pointerA] = encodeVer3(curr, prev)
			# end
			pointerA+=1
			while pointerA<=l && f[pointerA] <= 1
				pointerA += 1
			end
		end
		i = 0
		while i<l
			i+=1
			f[i] = floor(abs(f[i]))
		end
	end

# ╔═╡ 7ef5f9a7-de31-477a-8ee5-e699335836e5
function DT2WenboVer4!(f2d)
	i = size(f2d, 1)
	while i > 0
		DT1Wenbo!(@view(f2d[i, :]))
		i -= 1
	end
	j = size(f2d, 2)
	while j > 0
		DT2HelperVer4!(@view(f2d[:, j])) 
		j -= 1
	end
end

# ╔═╡ 57b7fa00-bffb-4564-8fac-f29253ec6f1f
 # 1.0     0.0     0.0     1.0      1.0      0.0      1.0
 # 0.0     1.0     4.0     9.0      4.0      1.0      0.0
 # 0.0     1.0     4.0     9.0     16.0     25.0     36.0
 # 1.0f10  1.0f10  1.0f10  1.0f10   1.0f10   1.0f10   1.0f10
 # 1.0f10  1.0f10  1.0f10  1.0f10   1.0f10   1.0f10   1.0f10
 # 1.0f10  1.0f10  1.0f10  1.0f10   1.0f10   1.0f10   1.0f10
 # 0.0     0.0     1.0     4.0      1.0      0.0      0.0
 # 1.0     0.0     0.0     0.0      1.0      0.0      1.0

# ╔═╡ 8a5666f7-ee15-4f8c-847f-9fb998a89c7e
# ╠═╡ disabled = true
#=╠═╡
@benchmark DT2($boolean_indicator($test2d4kresolu); D=$arg2d1)
  ╠═╡ =#

# ╔═╡ ef8cde04-6e7c-4b2a-864b-e3fee501d6a3
# ╠═╡ disabled = true
#=╠═╡
@benchmark DT2Wenbo($boolean_indicator($test2d4kresolu); D=$arg2d1, pointerA=1, pointerB=1)
  ╠═╡ =#

# ╔═╡ 2fd6d12f-5013-41ee-97b2-085d795ef3bd
@benchmark DT2Wenbo!($boolean_indicator($test2d4kresolu))

# ╔═╡ 88d27c48-7367-4455-9895-4e8c8b571a6d
# ╠═╡ disabled = true
#=╠═╡
@benchmark DT2WenboVer2!($boolean_indicator_Ver1($test2d4kresolu))
  ╠═╡ =#

# ╔═╡ 2f9e8f23-f47a-4a06-ba83-b63fc39708b9
@benchmark DT2WenboVer3!($boolean_indicator($test2d4kresolu))

# ╔═╡ d8e9c311-0962-4cb5-9214-43fa599fd8a4
@benchmark DT2WenboVer4!($boolean_indicator($test2d4kresolu))

# ╔═╡ 71072c67-6584-4fc7-ad4d-b3d362333562
md"""
## 3D
"""

# ╔═╡ 4731b952-c5aa-4c56-93ef-806e260eef73
function DT3(f; D=zeros(size(f)))
	for i = 1:size(f, 3)
	    D[:, :, i] = DT2(f[:, :, i]; D=D[:, :, i])
	end
	for i = 1:size(f, 1)
		for j = 1:size(f, 2)
	    	D[i, j, :] = DT1(D[i, j, :]; D= D[i, j, :])
		end
	end
	return D
end

# ╔═╡ f5ff25be-94d2-4e97-bb4d-df36abb4c0a8
function DT3Wenbo(f; D=zeros(size(f)), pointerA=1, pointerB=1)
	for i = 1:size(f, 3)
	    D[:, :, i] = DT2Wenbo(f[:, :, i]; D=D[:, :, i], pointerA=pointerA, pointerB=pointerB)
	end
	for i = 1:size(f, 1)
		for j = 1:size(f, 2)
	    	D[i, j, :] = DT2Helper(D[i, j, :], D[i, j, :])
		end
	end
	return D
end

# ╔═╡ 9daf9519-779a-47ec-b583-315fcab9fa54
function DT3WenboVer3!(f)
	for i = 1:size(f, 3)
	    DT2WenboVer3!(@view(f[:, :, i]))
	end
	for i = 1:size(f, 1)
		for j = 1:size(f, 2)
			DT2HelperVer3!(@view(f[i, j, :]))
		end
	end
end

# ╔═╡ d11b4fda-785c-4290-a713-4efda1223102
function DT3WenboVer4!(f)
	i = size(f, 3)
	while i > 0
	    DT2WenboVer4!(@view(f[:, :, i]))
		i -= 1
	end
	j = size(f, 1)
	kk = size(f, 2)
	while j > 0
		k = kk
		while k > 0
			DT2HelperVer4!(@view(f[j, k, :]))
			k -= 1
		end
		j -= 1
	end
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

# ╔═╡ 0ab25aa1-a6ac-43c6-bd90-e5cc67bcbbd7


# ╔═╡ 137ec483-53d4-4ce2-826b-af9d06fe9c62
@benchmark DT3($boolean_indicator($test3d1kresolu); D=$arg3d1)

# ╔═╡ a590e6c0-2f97-4866-bd0d-08b7758a347c
# ╠═╡ disabled = true
#=╠═╡
@benchmark DT3Wenbo($boolean_indicator($test3d1kresolu); D=$arg3d1, pointerA=1, pointerB=1)
  ╠═╡ =#

# ╔═╡ 67a91884-5169-4a32-aa56-35800a965e77
@benchmark DT3Wenbo!($boolean_indicator($test3d1kresolu))

# ╔═╡ 0e394677-e9ca-4211-b9fb-fdfd9165db9f
@benchmark DT3WenboVer3!($boolean_indicator($test3d1kresolu))

# ╔═╡ ab276915-8736-425b-860b-900aec89b697
@benchmark DT3WenboVer4!($boolean_indicator($test3d1kresolu))

# ╔═╡ 0e4d3711-f29b-4833-8a3e-a91e4375eec0
md"""
## Multi-Thded
"""

# ╔═╡ 9a0d9463-0002-4d0e-a368-3d7bba542e3d
nthreads = Threads.nthreads()

# ╔═╡ fb595c55-2995-462f-8a7f-982022370bd9
md"""
### 2D
"""

# ╔═╡ ae9df0c7-3dd4-4e11-8440-acba1faeb61b
function DT2Wenbo!(f2d, nthreads)
		Threads.@threads for i = 1:size(f2d, 1)
			DT1Wenbo!(@view(f2d[i, :]))
		end
		Threads.@threads for i = 1:size(f2d, 2)
			DT2Helper!(@view(f2d[:, i])) 
		end
	end

# ╔═╡ 8d2a10d8-bbea-4e7f-8a73-16548dd02e5e
 # 1.0     0.0     0.0     1.0      1.0      0.0      1.0
 # 0.0     1.0     4.0     9.0      4.0      1.0      0.0
 # 0.0     1.0     4.0     9.0     16.0     25.0     36.0
 # 1.0f10  1.0f10  1.0f10  1.0f10   1.0f10   1.0f10   1.0f10
 # 1.0f10  1.0f10  1.0f10  1.0f10   1.0f10   1.0f10   1.0f10
 # 1.0f10  1.0f10  1.0f10  1.0f10   1.0f10   1.0f10   1.0f10
 # 0.0     0.0     1.0     4.0      1.0      0.0      0.0
 # 1.0     0.0     0.0     0.0      1.0      0.0      1.0

 # 1.0  0.0  0.0   1.0  1.0  0.0  1.0
 # 0.0  1.0  1.0   2.0  2.0  1.0  0.0
 # 0.0  1.0  4.0   5.0  5.0  2.0  1.0
 # 1.0  2.0  5.0  10.0  8.0  5.0  4.0
 # 4.0  4.0  5.0   8.0  5.0  4.0  4.0
 # 1.0  1.0  2.0   4.0  2.0  1.0  1.0
 # 0.0  0.0  1.0   1.0  1.0  0.0  0.0
 # 1.0  0.0  0.0   0.0  1.0  0.0  1.0
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
	img2 = [
		0	1	1	0	0	1	0	
		1	0	0	0	0	0	1	
		1	0	0	0	0	0	0	
		0	0	0	0	0	0	0	
		0	0	0	0	0	0	0	
		0	0	0	0	0	0	0	
		1	1	0	0	0	1	1	
		0	1	1	1	0	1	0
	]
	test = boolean_indicator(img2)
	
	DT2Wenbo!(test)
	test
	
	# DT2(test)
end

# ╔═╡ 81472647-35a8-489e-a65d-b44586ddd179
begin
	function printErrorArea(input, output1, output2, x, y)
		x=min(x, size(input, 1)-4)
		x=max(x, 4)
		y=min(y, size(input, 2)-4)
		y=max(y, 4)
		for i = x-3:x+3
			for j = y-3:y+3
				if (input[i, j] == 0)				
					print("1");
				else							
					print("0");
				end

				print("\t");
			end
			println();
		end
		println();
		for i = x-3:x+3
			for j = y-3:y+3
				print(output1[i, j]);
				print("\t");
			end
			println();
		end
		println();
		for i = x-3:x+3
			for j = y-3:y+3
				print(output2[i, j]);
				print("\t");
			end
			println();
		end
		println();
	end
	function testDT2Wenbo!(size1, size2 , numCases)
		temp = 0
		for i in 1:numCases
			randomInput = rand([0, 1], size1, size2);
			testInput = boolean_indicator(randomInput);
			rslt = DT2(testInput);
			DT2Wenbo!(testInput);
			for j = 1:size1
				for k = 1:size2
					temp = max(temp, rslt[j, k])
					if (testInput[j, k] != rslt[j, k])
						printErrorArea(boolean_indicator(randomInput), rslt, testInput, j, k);
						return "Failed.";
					end 
				end
			end
		end
		print(temp)
		print("\n")
		return "DT2Wenbo! Passed.";
	end
	function testDT2WenboVer2!(size1, size2 , numCases)
		temp = 0
		for i in 1:numCases
			testInput = rand([0, 1], size1, size2);
			rslt = DT2(boolean_indicator(testInput));
			testInput = boolean_indicator_Ver1(testInput);
			DT2WenboVer2!(testInput);
			for j = 1:size1
				for k = 1:size2
					temp = max(temp, rslt[j, k])
					if (testInput[j, k] != rslt[j, k])
						printErrorArea(boolean_indicator(randomInput), rslt, testInput, j, k);
						return "Failed.";
					end 
				end
			end
		end
		println(temp)
		return "DT2WenboVer2! Passed.";
	end
	function testDT2WenboVer3!(size1, size2 , numCases)
		temp = 0
		for i in 1:numCases
			testInput = rand([0, 1], size1, size2);
			testInput = boolean_indicator(testInput);
			rslt = DT2(testInput);
			DT2WenboVer3!(testInput);
			for j = 1:size1
				for k = 1:size2
					temp = max(temp, rslt[j, k])
					if (testInput[j, k] != rslt[j, k])
						printErrorArea(testInput, rslt, testInput, j, k);
						return "Failed.";
					end 
				end
			end
		end
		println(temp)
		return "DT2WenboVer3! Passed.";
	end
	
	function testDT2WenboVer4!(size1, size2 , numCases)
		temp = 0
		for i in 1:numCases
			testInput = rand([0, 1], size1, size2);
			testInput = boolean_indicator(testInput);
			rslt = DT2(testInput);
			DT2WenboVer4!(testInput);
			for j = 1:size1
				for k = 1:size2
					temp = max(temp, rslt[j, k])
					if (testInput[j, k] != rslt[j, k])
						printErrorArea(testInput, rslt, testInput, j, k);
						return "Failed.";
					end 
				end
			end
		end
		println(temp)
		return "DT2WenboVer4! Passed.";
	end
	test2d4kresolu = rand([0, 1], 3840, 2160);
	arg2d1= zeros(size(test2d4kresolu));
	"Caught in 4k? So let's test in 4k resolution!" 
end

# ╔═╡ f80f3573-68a1-4732-be81-ffb8d4a45017
testDT2Wenbo!(3840, 2160, 50)

# ╔═╡ eacb2d56-3bf1-40cc-83d7-79af76f508be
testDT2WenboVer4!(3840, 2160, 50)

# ╔═╡ 803376a5-96fa-4a15-a354-6a98940768e0
function DT3Wenbo!(f)
	for i = 1:size(f, 3)
	    DT2Wenbo!(@view(f[:, :, i]))
	end
	for i = 1:size(f, 1)
		for j = 1:size(f, 2)
			DT2Helper!(@view(f[i, j, :]))
		end
	end
end

# ╔═╡ 8593e1bd-00e9-4096-830e-51633ad6b888
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
	test = boolean_indicator(img);
	DT2Wenbo!(test, nthreads);
	test
end

# ╔═╡ 8a73dd3d-94b3-48d7-8d86-b01614c856c0
# ╠═╡ disabled = true
#=╠═╡
@benchmark DT2Wenbo!($boolean_indicator($test2d4kresolu), nthreads)
  ╠═╡ =#

# ╔═╡ 36a60c88-2c32-4d21-80c3-ce965a49438b
# ╠═╡ disabled = true
#=╠═╡
md"""
### 3D
"""
  ╠═╡ =#

# ╔═╡ fb1ac75c-a2a7-4537-93a1-a036a8044a5b
function DT3Wenbo!(f, nthreads)
	Threads.@threads for i = 1:size(f, 3)
	    DT2WenboVer4!(@view(f[:, :, i]))
	end
	k = size(f, 1)
	 while k > 0
		Threads.@threads for j = 1:size(f, 2)
			DT2HelperVer4!(@view(f[k, j, :]))
		end
		k -= 1
	end
end

# ╔═╡ da9b8ae0-d446-47f6-8c03-2e145e83b7ed
begin
	function printErrorArea(input, output1, output2, x, y, z)
		x=min(x, size(input, 1)-4)
		x=max(x, 4)
		y=min(y, size(input, 2)-4)
		y=max(y, 4)
		z=min(z, size(input, 3)-4)
		z=max(z, 4)
		println("********* input *********\n");
		for i = x-3:x+3
			println(i);
			for j = y-3:y+3
				for k = z-3:z+3
					if (input[i, j, k] == 0)				
						print("1\t");
					else							
						print("0\t");
					end
				end
				println();
			end
			println();
		end
		println();
		println("********* correct *********\n");
		for i = x-3:x+3
			println(i);
			for j = y-3:y+3
				for k = z-3:z+3
					print(output1[i, j, k]);
					print("\t")
				end
				println();
			end
			println();
		end
		println();
		println("********* wrong *********\n");
		for i = x-3:x+3
			println(i);
			for j = y-3:y+3
				for k = z-3:z+3
					print(output2[i, j, k]);
					print("\t");
				end
				println();
			end
			println();
		end
		println();
	end
	
	function testDT3Wenbo!(size1, size2, size3, numCases)
		for i in 1:numCases
			randomInput = rand([0, 1], size1, size2, size3);
			testInput = boolean_indicator(randomInput);
			rslt = DT3(testInput);
			DT3Wenbo!(testInput);
			if (testInput != rslt)
				return "Failed.";
			end 
		end
		return "Passed.";
	end

	function testDT3WenboVer4!(size1, size2, size3, numCases)
		for i in 1:numCases
			randomInput = rand([0, 1], size1, size2, size3);
			testInput = boolean_indicator(randomInput);
			rslt = DT3(testInput);
			DT3WenboVer4!(testInput);
			
			for j = 1:size1
				for k = 1:size2
					for l = 1:size3
						if (testInput[j, k, l] != rslt[j, k, l])
							@printf "[%0.0f, %0.0f, %0.0f] : %0.2f, %0.2f" j k l testInput[j, k, l] rslt[j, k, l]
							printErrorArea(randomInput, rslt, testInput, j,k,l)
							return "Failed."
						end
					end
				end
			end
				
		end
		return "DT3WenboVer4 Passed."
	end
	test3d1kresolu = rand([0, 1], 1080, 1080, 150);
	arg3d1= zeros(size(test3d1kresolu));
	"testing 3D"
end

# ╔═╡ 3f1064a6-63e0-401b-8b57-7aaa4e097ccf
# ╠═╡ disabled = true
#=╠═╡
testDT3WenboVer4!(1080, 1080, 150,5)
  ╠═╡ =#

# ╔═╡ af6e4e5d-840d-4e9c-b091-d37b8c42f6ca
@benchmark DT3Wenbo!($boolean_indicator($test3d1kresolu), nthreads)

# ╔═╡ 406fcc98-5117-48d3-911f-91b350e6f9df
# ╠═╡ disabled = true
#=╠═╡
md"""
## GPU
"""
  ╠═╡ =#

# ╔═╡ 4c14ef1c-887a-44e3-84f6-9812bd433858
md"""
### 2D
"""

# ╔═╡ ceabaea4-17af-4fc2-8af4-b52c5fc886f5
md"""
### 3D
"""

# ╔═╡ Cell order:
# ╠═aec0d43c-16e0-4092-b81e-6dddbe41d3db
# ╠═b05b4228-fc30-4a74-8a99-f9b8d53aff83
# ╟─45c7f27a-b43c-4781-918a-51aebd273014
# ╟─1b7fee3a-1b11-411c-97a9-e6ede29ee56e
# ╠═45b90a66-956a-4b35-9e88-7621fd5c9c1a
# ╠═f6bc658b-dfd5-44e3-a6ff-26797729ff81
# ╠═a6c9dc03-74ce-4174-8fd9-6895d117aa4a
# ╠═d051233a-8363-4dd8-807f-2d273e55e29f
# ╠═6e2adb2c-93f4-4a19-b93c-75b75a9116f3
# ╠═f6631dd6-cb06-4c84-a5db-38ad6af64b68
# ╠═f873e7c2-7dac-4925-b6b5-5887b461418b
# ╠═0c569423-c87f-4139-a2e6-2e1a897d5f39
# ╟─8aab09c7-07fe-4691-9e1d-35e2e215bc5d
# ╟─e75c9e76-3634-4bde-abb8-5b4827eb089e
# ╟─17d3777b-69d4-40c1-84c1-2bc6a737d52a
# ╟─258346da-e792-11ec-06dc-5906e95d24e2
# ╟─098b8119-26e8-4002-84c4-c8478d3c5019
# ╟─93de6d70-46ff-414f-9185-72975b252cbe
# ╟─5dd214c2-91ee-43cf-9e1f-97dd64fbd383
# ╟─79041c45-d062-4b0d-8643-aca2a6b6ccf1
# ╟─230a3b21-01d3-4728-a07b-8382aa1e5db1
# ╟─debda360-c879-41cc-a8ed-281b75b33934
# ╟─fcea1ad8-3fb1-4e82-b37f-80bed8a11df2
# ╟─c3ab6a19-873f-41c1-b728-caf89c52bf25
# ╟─722a98fa-7c00-46ad-b0d6-ba83d258204c
# ╠═37124f17-dee2-40b0-bbb1-b5946c7efa48
# ╠═162718f4-40c7-4bc8-9f8c-f9f4ae56573d
# ╠═2280361b-068d-4d5e-ae30-7e66c42ee665
# ╟─03d7270c-58ee-4ce2-a67d-262bc62c9478
# ╟─c8e739c0-940d-47b4-be2e-b0345a7e362f
# ╟─ea2e8c7e-23fe-4277-bb4d-ae56123163ab
# ╟─0469116a-7e10-44c5-801d-f5ce683dc486
# ╟─0cca3714-b73a-4f6e-b953-e01d392bdc91
# ╠═30ae8059-4b4e-4026-966e-7d6320912b5c
# ╠═7a1a556f-fa9e-4cec-88e8-5c74c10e0c4e
# ╠═5f9287b3-c7c3-4c0d-bf3b-d3161c4bb249
# ╠═cc775fe3-563d-46ff-8adc-4164dd0e9ea2
# ╠═c0a2399a-147a-41ca-8a27-4bc6a70dca52
# ╟─2d02a6fd-7a23-4790-b8da-7b7150482a85
# ╟─29cb3c08-a702-4564-ad41-664f312913b0
# ╟─2d7da285-7078-4567-8cdd-a8cefd737f57
# ╟─535c4318-d29e-4640-9105-66b16d13bf88
# ╟─56a72a97-1505-4927-9106-9e4e0111dba9
# ╟─0509a408-2dbc-483f-bd5f-b9832268845a
# ╟─9eef234c-f269-4437-9e0e-299447cd1d10
# ╟─bce0d88f-4384-4b22-8b19-a3980e81212b
# ╟─40c9bdbd-6898-4919-9261-671ff7702296
# ╟─416e2c45-2f56-4136-bfe5-e411784e9829
# ╟─ec92e0e4-95f9-4dfd-b62a-2387493c426a
# ╟─24146d99-44f0-486d-b8ab-5d0f1c54e26b
# ╟─7f835715-f871-4abd-b8fa-d383a9a52b3c
# ╟─3eb49acc-ae0f-4ba4-aa9e-a8899d8d4960
# ╟─6c041277-0246-4e22-b296-2a4a8160d247
# ╟─7ef5f9a7-de31-477a-8ee5-e699335836e5
# ╟─57b7fa00-bffb-4564-8fac-f29253ec6f1f
# ╟─8d2a10d8-bbea-4e7f-8a73-16548dd02e5e
# ╠═81472647-35a8-489e-a65d-b44586ddd179
# ╠═f80f3573-68a1-4732-be81-ffb8d4a45017
# ╠═eacb2d56-3bf1-40cc-83d7-79af76f508be
# ╠═8a5666f7-ee15-4f8c-847f-9fb998a89c7e
# ╟─ef8cde04-6e7c-4b2a-864b-e3fee501d6a3
# ╠═2fd6d12f-5013-41ee-97b2-085d795ef3bd
# ╟─88d27c48-7367-4455-9895-4e8c8b571a6d
# ╠═2f9e8f23-f47a-4a06-ba83-b63fc39708b9
# ╠═d8e9c311-0962-4cb5-9214-43fa599fd8a4
# ╟─71072c67-6584-4fc7-ad4d-b3d362333562
# ╟─4731b952-c5aa-4c56-93ef-806e260eef73
# ╟─f5ff25be-94d2-4e97-bb4d-df36abb4c0a8
# ╟─803376a5-96fa-4a15-a354-6a98940768e0
# ╟─9daf9519-779a-47ec-b583-315fcab9fa54
# ╟─d11b4fda-785c-4290-a713-4efda1223102
# ╟─aeabedd0-5719-46f2-9715-7b21d8e5bf3d
# ╠═0ab25aa1-a6ac-43c6-bd90-e5cc67bcbbd7
# ╠═da9b8ae0-d446-47f6-8c03-2e145e83b7ed
# ╠═3f1064a6-63e0-401b-8b57-7aaa4e097ccf
# ╠═137ec483-53d4-4ce2-826b-af9d06fe9c62
# ╠═a590e6c0-2f97-4866-bd0d-08b7758a347c
# ╠═67a91884-5169-4a32-aa56-35800a965e77
# ╠═0e394677-e9ca-4211-b9fb-fdfd9165db9f
# ╠═ab276915-8736-425b-860b-900aec89b697
# ╠═0e4d3711-f29b-4833-8a3e-a91e4375eec0
# ╠═9a0d9463-0002-4d0e-a368-3d7bba542e3d
# ╠═fb595c55-2995-462f-8a7f-982022370bd9
# ╟─ae9df0c7-3dd4-4e11-8440-acba1faeb61b
# ╟─8593e1bd-00e9-4096-830e-51633ad6b888
# ╠═8a73dd3d-94b3-48d7-8d86-b01614c856c0
# ╠═36a60c88-2c32-4d21-80c3-ce965a49438b
# ╠═fb1ac75c-a2a7-4537-93a1-a036a8044a5b
# ╠═af6e4e5d-840d-4e9c-b091-d37b8c42f6ca
# ╠═406fcc98-5117-48d3-911f-91b350e6f9df
# ╠═4c14ef1c-887a-44e3-84f6-9812bd433858
# ╠═ceabaea4-17af-4fc2-8af4-b52c5fc886f5
