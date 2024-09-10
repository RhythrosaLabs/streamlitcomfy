class Preset:
    def __init__(self, name, node_sequence):
        self.name = name
        self.node_sequence = node_sequence

# Define the available nodes (ensure these match the nodes from the main app)
def get_available_nodes():
    return {
        "flux": AINode("flux", "Flux Schnell", "black-forest-labs/flux-schnell", "text", "image"),
        "sdxl": AINode("sdxl", "Stable Diffusion XL", "stability-ai/sdxl:a00d0b7dcbb9c3fbb34ba87d2d5b46c56969c84a628bf778a7fdaec30b1b99c5", "text", "image"),
        "upscale": AINode("upscale", "Image Upscaling", "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b", "image", "image"),
        "remove-bg": AINode("remove-bg", "Remove Background", "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003", "image", "image"),
        "video": AINode("video", "Video Generation", "anotherjesse/zeroscope-v2-xl:9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351", "text", "video"),
    }

# Define some preset workflows
def get_presets():
    nodes = get_available_nodes()
    return [
        Preset("Image Generate -> Upscale -> Remove Background", [nodes["flux"], nodes["upscale"], nodes["remove-bg"]]),
        Preset("Image Generate -> Remove Background -> Video", [nodes["flux"], nodes["remove-bg"], nodes["video"]]),
        Preset("Stable Diffusion XL -> Upscale -> Remove Background", [nodes["sdxl"], nodes["upscale"], nodes["remove-bg"]]),
    ]
