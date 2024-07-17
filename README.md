# Diffusers Core ML

## Tested on

|      Device       | macOS  | Python | coremltools |
|:-----------------:|:------:|:------:|:-----------:|
| MacBook 15 M3 16G |   14.5 |  3.9   |     7.2     |

## Performance

|        Model         | Quantization | Compute Unit | Latency(s) |
|:--------------------:|:------------:|:------------:|:----------:|
| SDXL Lightning 4step |    6bits     | CPU_AND_GPU  |     15     |

## Installation

```
pip3 install git+https://github.com/digitalbrain79/transformers_coreml.git
pip3 install git+https://github.com/digitalbrain79/diffusers_coreml.git
```

## Example
### Text to Image

```py
from diffusers import (
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLPipeline
)

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "digitalbrain79/sdxl_lightning_4step_coreml_6bits_compiled",
    use_safetensors=False,
    low_cpu_mem_usage=False
)

pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing="trailing"
)

image = pipeline(
    prompt="a photo of an astronaut riding a horse on mars",
    num_inference_steps=4,
    guidance_scale=0
).images[0]
```

<img src="assets/text_to_image.png" width="384">

### Text to Image

```py
from diffusers import (
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLImg2ImgPipeline
)
from diffusers.utils import load_image

pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "digitalbrain79/sdxl_lightning_4step_coreml_6bits_compiled",
    use_safetensors=False,
    low_cpu_mem_usage=False
)

pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing="trailing"
)

url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"
init_image = load_image(url).convert("RGB")
image = pipeline(
    prompt="an astronaut riding a horse on mars, anime style",
    image=init_image,
    strength=0.9,
    num_inference_steps=4,
    guidance_scale=0
).images[0]
```

<img src="assets/image_to_image.png" width="384">
