### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 514a8cf8-15ff-4896-8905-3afcaa8b4d3a
begin
    let
        using Pkg
        # Pkg.activate(raw"G:\molloi-lab\project-distance-transforms\julia_env_wenboPC")
        # Pkg.instantiate()

        cd(raw"G:\桌面\test_julia")
        Pkg.activate(mktempdir())
        Pkg.instantiate()
        #]Pkg.Registry.update()
        Pkg.add("PlutoUI")
        Pkg.add("Tar")
        Pkg.add("MLDataPattern")
        Pkg.add("Glob")
        Pkg.add("NIfTI")
        Pkg.add("DataAugmentation")
        Pkg.add("CairoMakie")
        Pkg.add("ImageCore")
        Pkg.add("DataLoaders")
        Pkg.add("CUDA")
        Pkg.add(Pkg.PackageSpec(;name="FastAI", version="0.2.0"))
        Pkg.add(url = "https://github.com/FluxML/Metalhead.jl")
        Pkg.add("StaticArrays")
        

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
    using DataLoaders
    using CUDA
    using FastAI
    using Metalhead
    using StaticArrays
    using torch
end

# ╔═╡ 66b8f955-7d22-421e-bbef-5180cedaa4be


# ╔═╡ 7c4d8165-dd39-4043-bafa-56626c4fdde6


# ╔═╡ 48f904bc-af7c-46b0-84a6-f0f45fcffd7d
TableOfContents()

# ╔═╡ 58185eb7-6205-49a4-9b92-6881eeb6020c
md"""
## GPU
"""

# ╔═╡ 9dee5ba8-ea30-429a-9466-7b4dfabe5671
@show CUDA.functional()

# ╔═╡ fc34ef2e-7403-4733-b477-b04f705a9b8e
@show CUDA.device()

# ╔═╡ 5f77cbb8-26a4-4287-b4a3-c9c1158f8d1f
md"""
## Load data
Part of the [Medical Decathlon Dataset](http://medicaldecathlon.com/)
"""

# ╔═╡ 7ec89eaf-7eca-47d9-90fc-c6b9fa241f2e
data_dir = raw"G:\桌面\DT Data\Task02_Heart"

# ╔═╡ 4294429e-e29e-4986-b0d7-66d8a1d7d3a5
function loadfn_label(p)
    a = NIfTI.niread(string(p)).raw
    convert_a = convert(Array{UInt8}, a)
    convert_a = convert_a .+ 1
    return convert_a
end

# ╔═╡ 838a17df-7479-42c2-a2c5-ce626a4016fe
function loadfn_image(p)
    a = NIfTI.niread(string(p)).raw
    convert_a = convert(Array{Float32}, a)
    convert_a = convert_a / max(convert_a...)
    return convert_a
end

# ╔═╡ 11d5a9b4-fdc3-48e4-a6ad-f7dec5fabb8b
begin
    niftidata_image(dir) = mapobs(loadfn_image, Glob.glob("*.nii*", dir))
    niftidata_label(dir) =  mapobs(loadfn_label, Glob.glob("*.nii*", dir))
    data = (
        niftidata_image(joinpath(data_dir, "imagesTr")),
        niftidata_label(joinpath(data_dir, "labelsTr")),
    )
end

# ╔═╡ 778cc0f9-127c-4d4b-a7de-906dfcc29cae
train_files, val_files = MLDataPattern.splitobs(data, 0.8)

# ╔═╡ 349d843a-4a5f-44d7-9371-38c140b9972d
md"""
## Create learning method
"""

# ╔═╡ 019e666e-e2e4-4e0b-a225-b346c7c70939
struct ImageSegmentationSimple <: DLPipelines.LearningMethod
    imagesize
end

# ╔═╡ f7274fa9-8231-44fd-8d00-1c7ab7fc855c
image_size = (96, 96, 96)

# ╔═╡ 9ac18928-9fe4-46ed-ab9c-916791739157
method = ImageSegmentationSimple(image_size)

# ╔═╡ edf2b37a-2775-44c0-8d2e-5e2350b454c4
md"""
### Set up `encode` pipelines
"""

# ╔═╡ 7cf0cc6b-8ff9-4198-9646-9d0787e1013d
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

# ╔═╡ f2a92ff7-0f94-44d9-aba7-4b7db9fa4a56
begin
	methoddata_train = DLPipelines.MethodDataset(train_files, method, Training())
	methoddata_valid = DLPipelines.MethodDataset(val_files, method, Validation())
end

# ╔═╡ b8b18728-cb5a-445a-809c-986e3965aad3
begin
    x, y = MLDataPattern.getobs(methoddata_valid, 1)
    @assert size(x) == (image_size..., 1)
    @assert size(y) == (image_size..., 2)
end

# ╔═╡ bfe051e5-66a3-4b4b-b446-48195d8f3868
md"""
## Visualize
"""

# ╔═╡ 6d1e9ca8-da31-4143-b46f-44b459a2cfc3
@show size(x)

# ╔═╡ 792a709c-6740-4fc0-b2a3-43ad2ae8035a
@show size(y)

# ╔═╡ 08be9516-a44c-4fe5-ac46-f586600d586a
@bind b PlutoUI.Slider(1:size(x)[3], default=50, show_value=true)

# ╔═╡ e2670a05-2c09-4bd2-b40c-0cc46fef2344
heatmap(x[:, :, b, 1], colormap=:grays)

# ╔═╡ 427ff0c3-d773-4099-a483-bb8f7d4b8ba1
heatmap(y[:, :, b, 1], colormap=:grays)

# ╔═╡ 616aa780-cbcf-4bf2-b6c5-a984f2530482
md"""
## Dataloader
"""

# ╔═╡ bda7309e-ae97-4ac0-96b6-36a776e9215e
begin
    train_loader = DataLoaders.DataLoader(methoddata_train, 1)
    val_loader = DataLoaders.DataLoader(methoddata_valid, 4)
end

# ╔═╡ 3da0e60e-0196-4538-ab08-49f21b46679a
train_loader

# ╔═╡ e70816b8-4597-4a54-b4f7-735880df6132
val_loader

# ╔═╡ ff962aea-2501-42f5-90bc-72f29deb42af
md"""
## Model
"""

# ╔═╡ 5987b24a-a1da-4d13-89d1-c854de2c3ba0
begin
    # 3D layer utilities
    conv = (stride, in, out) -> Conv((3, 3, 3), in=>out, stride=stride, pad=(1, 1, 1))
    tran = (stride, in, out) -> ConvTranspose((4, 4, 4), in=>out, stride=stride, pad=1)

    conv1 = (in, out) -> Chain(conv(1, in, out), BatchNorm(out), x -> leakyrelu.(x))
    conv2 = (in, out) -> Chain(conv(2, in, out), BatchNorm(out), x -> leakyrelu.(x))
    tran2 = (in, out) -> Chain(tran(2, in, out), BatchNorm(out), x -> leakyrelu.(x))
end

# ╔═╡ f451e763-d7d0-4de7-a1f3-76776a194022
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

# ╔═╡ ad342428-920a-4414-a4cc-cab281a681dc
model = unet3D(1, 2) |> gpu;

# ╔═╡ d073cbac-18c8-4682-a508-307a619e84fc
md"""
## Helper functions
"""

# ╔═╡ 5726b8bd-e1e7-44eb-8872-a4a8d26be0f9
function dice_metric(ŷ, y)
    dice = 2 * sum(ŷ .& y) / (sum(ŷ) + sum(y))
    return dice
end

# ╔═╡ 7b974fc5-7fc3-45e7-bf6c-48ea4f8eff16
function as_discrete(array, logit_threshold)
    array = array .>= logit_threshold
    return array
end

# ╔═╡ b87d764e-dd31-48b7-bcf1-0538ef5c5e6a
md"""
## Loss functions
"""

# ╔═╡ 6dee15db-5cf9-4bfb-ab42-48f1c2777c04
function dice_loss(ŷ, y)
    ϵ = 1e-5
    return loss = 1 - ((2 * sum(ŷ .* y) + ϵ) / (sum(ŷ .* ŷ) + sum(y .* y) + ϵ))
end

# ╔═╡ bdc1ca32-4847-440f-8a44-98bf7e822803
function hd_loss(ŷ, y, ŷ_dtm, y_dtm)
    M = (ŷ .- y) .^ 2 .* (ŷ_dtm .^ 2 .+ y_dtm .^ 2)
    return loss = mean(M)
end

# ╔═╡ aafb8a3f-6e6c-4475-92db-550eb6999741
md"""
## Training
"""

# ╔═╡ 3d0f1399-54c3-45b6-840a-8e287513bbe7
ps = Flux.params(model);

# ╔═╡ 06061dc5-eae9-47cb-99aa-d521c5cd37dd
loss_function = dice_loss

# ╔═╡ 3a13a9a9-380a-4613-b137-99fdef8cb92f
optimizer = Flux.ADAM(0.01)

# ╔═╡ 36501d5d-d29a-4281-9e75-cebc97bb68cd
begin
	max_epochs = 2
	val_interval = 1
	epoch_loss_values = []
	val_epoch_loss_values = []
	dice_metric_values = []
end

# ╔═╡ 34905136-e11c-4d15-9f63-57cc66c4e514
for (xs, ys) in val_loader
	@show size(xs)
	@show size(ys)
end

# ╔═╡ ad1bf410-f187-4811-b752-09252e1b3a32
begin
	for epoch in 1:max_epochs
		step = 0
		@show epoch
		
		# Loop through training data
		for (xs, ys) in train_loader
			xs, ys = xs |> gpu, ys |> gpu
			step += 1
			@show step
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
				@show val_step

				local val_ŷs = model(val_xs)
				local val_loss = loss_function(val_ŷs[:, :, :, 2, :], val_ys[:, :, :, 2, :])
				val_ŷs, val_ys = as_discrete(val_ŷs, 0.5), as_discrete(val_ys, 0.5)
			end
		end
	end
end
