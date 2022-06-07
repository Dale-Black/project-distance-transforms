### A Pluto.jl notebook ###
# v0.18.1

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

# ╔═╡ 2eae761f-cbc3-4f29-b1d3-238bdf8b8a8e
begin
    let
        using Pkg
		# cd("C:/Users/Test/centerline-extraction")
		Pkg.activate(mktempdir())
		Pkg.Registry.update()
        Pkg.add("PlutoUI")
        Pkg.add("Tar")
        Pkg.add("MLDataPattern")
        Pkg.add("Glob")
        Pkg.add("NIfTI")
        Pkg.add("DataAugmentation")
        Pkg.add("CairoMakie")
        Pkg.add("ImageCore")
		Pkg.add("DLPipelines")
        Pkg.add("DataLoaders")
        Pkg.add("CUDA")
        Pkg.add("FastAI")
    end

    using PlutoUI
    using Tar
    using MLDataPattern
    using Glob
    using NIfTI
    using DataAugmentation
    using DataAugmentation: OneHot, Image
    using CairoMakie
    using ImageCore
	using DLPipelines
    using DataLoaders
    using CUDA
    using FastAI
end

# ╔═╡ ef3eb363-b289-45ce-8e2b-4b5fe036510c
TableOfContents()

# ╔═╡ e3a2aa0b-e631-4590-9106-15c92252abc8
md"""
## Load data
Part of the [Medical Decathlon Dataset](http://medicaldecathlon.com/)
"""

# ╔═╡ 9febfd0f-18d2-4f20-8c39-45d5ab31c803
data_dir = "/Users/daleblack/Google Drive/Datasets/Task02_Heart"

# ╔═╡ 54106b02-0a26-4b5c-bd07-1263192a8c86
function loadfn_label(p)
    a = NIfTI.niread(string(p)).raw
    convert_a = convert(Array{UInt8}, a)
    convert_a = convert_a .+ 1
    return convert_a
end

# ╔═╡ 7be75367-5035-42c5-a3e2-2ad7c2b62cf7
function loadfn_image(p)
    a = NIfTI.niread(string(p)).raw
    convert_a = convert(Array{Float32}, a)
    convert_a = convert_a / max(convert_a...)
    return convert_a
end

# ╔═╡ 7f5c87d8-b822-4ec6-90c4-2ca93308cbf4
begin
    niftidata_image(dir) = mapobs(loadfn_image, Glob.glob("*.nii*", dir))
    niftidata_label(dir) = mapobs(loadfn_label, Glob.glob("*.nii*", dir))
    data = (
        niftidata_image(joinpath(data_dir, "imagesTr")),
        niftidata_label(joinpath(data_dir, "labelsTr")),
    )
end

# ╔═╡ 1f483650-5c30-4e81-b7b8-8f90f7521658
train_files, val_files = MLDataPattern.splitobs(data, 0.8)

# ╔═╡ 60c6dd83-16f2-4c9e-aa4f-a61cb3d4d637
md"""
## Create learning method
"""

# ╔═╡ 613e31f4-ac2b-406d-a58a-6a93be05ebac
struct ImageSegmentationSimple <: DLPipelines.LearningMethod
    imagesize
end

# ╔═╡ bd2fda81-032d-402d-b38e-53bd20e14d84
image_size = (64, 64, 64)

# ╔═╡ fd7f5cac-b25a-4fcf-aa77-8d35ec5c2d65
method = ImageSegmentationSimple(image_size)

# ╔═╡ c6afb211-3c36-497f-8a3a-b0f9cc17b2e4
md"""
### Set up `encode` pipelines
"""

# ╔═╡ 7b663349-f498-4e85-8ab0-befd5e582ddb
begin
  function DLPipelines.encode(
          method::ImageSegmentationSimple,
          context::Training,
          (image, target)::Union{Tuple, NamedTuple}
          )

      tfm_proj = RandomResizeCrop(method.imagesize)
      tfm_im = DataAugmentation.compose(
			ImageToTensor(),
			NormalizeIntensity()
          )
      tfm_mask = OneHot()

      items = Image(Gray.(image)), MaskMulti(target)
      item_im, item_mask = apply(tfm_proj, (items))

      return itemdata(apply(tfm_im, item_im)), itemdata(apply(tfm_mask, item_mask))
  end

  function DLPipelines.encode(
          method::ImageSegmentationSimple,
          context::Validation,
          (image, target)::Union{Tuple, NamedTuple}
          )

      tfm_proj = CenterResizeCrop(method.imagesize)
      tfm_im = DataAugmentation.compose(
          ImageToTensor(),
          NormalizeIntensity()
          )
      tfm_mask = OneHot()

      items = Image(Gray.(image)), MaskMulti(target)
      item_im, item_mask = apply(tfm_proj, (items))

      return itemdata(apply(tfm_im, item_im)), itemdata(apply(tfm_mask, item_mask))
  end
end

# ╔═╡ a4c1d6ad-7b1f-4a44-bfa2-77aa276ad4eb
begin
	methoddata_train = DLPipelines.MethodDataset(train_files, method, Training())
	methoddata_valid = DLPipelines.MethodDataset(val_files, method, Validation())
end

# ╔═╡ c5672f5b-ba85-4c30-afb9-1951e9c51110
begin
    x, y = MLDataPattern.getobs(methoddata_valid, 1)
    @assert size(x) == (image_size..., 1)
    @assert size(y) == (image_size..., 2)
end

# ╔═╡ 273e99f1-47ec-4c57-9357-2565fd0d568b
md"""
## Visualize
"""

# ╔═╡ c17d636e-ca60-42b4-a3ca-027074376605
size(x)

# ╔═╡ 37098afc-bbba-4ad7-8eb0-3bbf054ec3ae
size(y)

# ╔═╡ 645584ac-d49d-4edd-80a3-bcf220cda411
@bind b PlutoUI.Slider(1:size(x)[3], default=50, show_value=true)

# ╔═╡ 95a0392c-0898-4062-9152-3d65aba16977
heatmap(x[:, :, b, 1], colormap=:grays)

# ╔═╡ a7084f2c-3e80-449a-88b2-5a594252b802
heatmap(y[:, :, b, 3], colormap=:grays)

# ╔═╡ 64d53082-4d06-4bc3-886d-c8fccdd31830
md"""
## Dataloader
"""

# ╔═╡ 561702e1-879a-424c-98c9-c3133bbb0568
begin
    train_loader = DataLoaders.DataLoader(methoddata_train, 1)
    val_loader = DataLoaders.DataLoader(methoddata_valid, 4)
end

# ╔═╡ f8d1bb3d-2a14-4aaf-9ab5-f8fef90175d7
train_loader

# ╔═╡ 307bf825-66d5-4700-ba6f-42ab27258782
val_loader

# ╔═╡ ddc151f7-6b54-4e57-9dce-d442bdc749b6
md"""
## Model
"""

# ╔═╡ f589f614-5625-41f3-a1cc-4a280d9ea055
begin
    # 3D layer utilities
    conv = (stride, in, out) -> Conv((3, 3, 3), in=>out, stride=stride, pad=(1, 1, 1))
    tran = (stride, in, out) -> ConvTranspose((4, 4, 4), in=>out, stride=stride, pad=1)

    conv1 = (in, out) -> Chain(conv(1, in, out), BatchNorm(out), x -> leakyrelu.(x))
    conv2 = (in, out) -> Chain(conv(2, in, out), BatchNorm(out), x -> leakyrelu.(x))
    tran2 = (in, out) -> Chain(tran(2, in, out), BatchNorm(out), x -> leakyrelu.(x))
end

# ╔═╡ 59566701-4070-4039-812a-3aa6468b5f17
function unet3D(in_chs, lbl_chs)
    # Contracting layers
    l1 = Chain(conv1(in_chs, 4))
    l2 = Chain(l1, conv1(4, 4), conv2(4, 16))
    l3 = Chain(l2, conv1(16, 16), conv2(16, 32))
    l4 = Chain(l3, conv1(32, 32), conv2(32, 64))
    l5 = Chain(l4, conv1(64, 64), conv2(64, 128))

    # Expanding layers
    l6 = Chain(l5, tran2(128, 64), conv1(64, 64))
    l7 = Chain(Parallel(+, l6, l4), tran2(64, 32), conv1(32, 32))       # Residual connection between l6 & l4
    l8 = Chain(Parallel(+, l7, l3), tran2(32, 16), conv1(16, 16))       # Residual connection between l7 & l3
    l9 = Chain(Parallel(+, l8, l2), tran2(16, 4), conv1(4, 4))          # Residual connection between l8 & l2
    l10 = Chain(l9, conv1(4, lbl_chs))
end

# ╔═╡ bbba9b2c-cd4b-42cf-9925-7a5d23b7a268
model = unet3D(1, 2) |> gpu;

# ╔═╡ 7aa665a8-5064-4d06-92a6-abd253c249ed
md"""
## Helper functions
"""

# ╔═╡ 06f1807e-e364-4a3f-9afd-dbbde4430861
function dice_metric(ŷ, y)
    dice = 2 * sum(ŷ .& y) / (sum(ŷ) + sum(y))
    return dice
end

# ╔═╡ 2a1374b7-7b6f-451d-91ab-27ed91688b85
function as_discrete(array, logit_threshold)
    array = array .>= logit_threshold
    return array
end

# ╔═╡ 658ffe56-a70e-4f44-9fdf-46960490f679
md"""
## Loss functions
"""

# ╔═╡ f2eaed16-b341-4124-866c-c4188cf17fae
function dice_loss(ŷ, y)
    ϵ = 1e-5
    return loss = 1 - ((2 * sum(ŷ .* y) + ϵ) / (sum(ŷ .* ŷ) + sum(y .* y) + ϵ))
end

# ╔═╡ ebaeb819-8af4-4e20-a679-575e43dcc126
function hd_loss(ŷ, y, ŷ_dtm, y_dtm)
    M = (ŷ .- y) .^ 2 .* (ŷ_dtm .^ 2 .+ y_dtm .^ 2)
    return loss = mean(M)
end

# ╔═╡ dd0b8db6-6013-4dc9-af25-b4e46d9148fa
md"""
## Training
"""

# ╔═╡ 7a989e6a-4380-4a6e-acc6-b369784efd09
ps = Flux.params(model);

# ╔═╡ 6b281609-dbc4-43d9-acc0-8e519db786ad
loss_function = dice_loss

# ╔═╡ 9ddae6ce-b100-4f57-beec-081b7ff5c68a
optimizer = Flux.ADAM(0.01)

# ╔═╡ fffeee91-4e55-4bb3-abd1-420f853f5452
begin
	max_epochs = 2
	val_interval = 1
	epoch_loss_values = []
	val_epoch_loss_values = []
	dice_metric_values = []
end

# ╔═╡ 05b2394a-54a1-42f8-bd62-3bb092994368
for (xs, ys) in val_loader
	@info size(xs)
	@info size(ys)
end

# ╔═╡ 72443787-7500-45ab-b116-4572aeae1996
begin
	for epoch in 1:max_epochs
		step = 0
		@info epoch
		
		# Loop through training data
		for (xs, ys) in train_loader
			xs, ys = xs |> gpu, ys |> gpu
			step += 1
			@info step
			gs = Flux.gradient(ps) do
				ŷs = model(xs)
				loss = loss_function(ŷs[:, :, :, 2, :], ys[:, :, :, 2, :])
				return loss
			end
			Flux.update!(optimizer, ps, gs)
		end

		# Loop through validation data
		if (epoch + 1) % val_interval == 0
			val_step = 0
			for (val_xs, val_ys) in val_loader
				val_xs, val_ys = val_xs |> gpu, val_ys |> gpu 
				val_step += 1
				@info val_step

				local val_ŷs = model(val_xs)
				local val_loss = loss_function(val_ŷs[:, :, :, 2, :], val_ys[:, :, :, 2, :])
				val_ŷs, val_ys = as_discrete(val_ŷs, 0.5), as_discrete(val_ys, 0.5)
			end
		end
	end
end

# ╔═╡ Cell order:
# ╠═2eae761f-cbc3-4f29-b1d3-238bdf8b8a8e
# ╠═ef3eb363-b289-45ce-8e2b-4b5fe036510c
# ╟─e3a2aa0b-e631-4590-9106-15c92252abc8
# ╠═9febfd0f-18d2-4f20-8c39-45d5ab31c803
# ╠═54106b02-0a26-4b5c-bd07-1263192a8c86
# ╠═7be75367-5035-42c5-a3e2-2ad7c2b62cf7
# ╠═7f5c87d8-b822-4ec6-90c4-2ca93308cbf4
# ╠═1f483650-5c30-4e81-b7b8-8f90f7521658
# ╟─60c6dd83-16f2-4c9e-aa4f-a61cb3d4d637
# ╠═613e31f4-ac2b-406d-a58a-6a93be05ebac
# ╠═bd2fda81-032d-402d-b38e-53bd20e14d84
# ╠═fd7f5cac-b25a-4fcf-aa77-8d35ec5c2d65
# ╟─c6afb211-3c36-497f-8a3a-b0f9cc17b2e4
# ╠═7b663349-f498-4e85-8ab0-befd5e582ddb
# ╠═a4c1d6ad-7b1f-4a44-bfa2-77aa276ad4eb
# ╠═c5672f5b-ba85-4c30-afb9-1951e9c51110
# ╠═273e99f1-47ec-4c57-9357-2565fd0d568b
# ╠═c17d636e-ca60-42b4-a3ca-027074376605
# ╠═37098afc-bbba-4ad7-8eb0-3bbf054ec3ae
# ╠═645584ac-d49d-4edd-80a3-bcf220cda411
# ╠═95a0392c-0898-4062-9152-3d65aba16977
# ╠═a7084f2c-3e80-449a-88b2-5a594252b802
# ╟─64d53082-4d06-4bc3-886d-c8fccdd31830
# ╠═561702e1-879a-424c-98c9-c3133bbb0568
# ╠═f8d1bb3d-2a14-4aaf-9ab5-f8fef90175d7
# ╠═307bf825-66d5-4700-ba6f-42ab27258782
# ╠═ddc151f7-6b54-4e57-9dce-d442bdc749b6
# ╠═f589f614-5625-41f3-a1cc-4a280d9ea055
# ╠═59566701-4070-4039-812a-3aa6468b5f17
# ╠═bbba9b2c-cd4b-42cf-9925-7a5d23b7a268
# ╠═7aa665a8-5064-4d06-92a6-abd253c249ed
# ╠═06f1807e-e364-4a3f-9afd-dbbde4430861
# ╠═2a1374b7-7b6f-451d-91ab-27ed91688b85
# ╠═658ffe56-a70e-4f44-9fdf-46960490f679
# ╠═f2eaed16-b341-4124-866c-c4188cf17fae
# ╠═ebaeb819-8af4-4e20-a679-575e43dcc126
# ╠═dd0b8db6-6013-4dc9-af25-b4e46d9148fa
# ╠═7a989e6a-4380-4a6e-acc6-b369784efd09
# ╠═6b281609-dbc4-43d9-acc0-8e519db786ad
# ╠═9ddae6ce-b100-4f57-beec-081b7ff5c68a
# ╠═fffeee91-4e55-4bb3-abd1-420f853f5452
# ╠═05b2394a-54a1-42f8-bd62-3bb092994368
# ╠═72443787-7500-45ab-b116-4572aeae1996
