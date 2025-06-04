STYLES = {
    "None": "{prompt}",
    "Cinematic": "cinematic still {prompt}. emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
    "3D Model": "professional 3d model {prompt}. octane render, highly detailed, volumetric, dramatic lighting",
    "Anime": "anime artwork {prompt}. anime style, key visual, vibrant, studio anime,  highly detailed",
    "Digital Art": "concept art {prompt}. digital artwork, illustrative, painterly, matte painting, highly detailed",
    "Photographic": "cinematic photo {prompt}. 35mm photograph, film, bokeh, professional, 4k, highly detailed",
    "Pixel art": "pixel-art {prompt}. low-res, blocky, pixel art style, 8-bit graphics",
    "Fantasy art": "ethereal fantasy concept art of  {prompt}. magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
    "Neonpunk": "neonpunk style {prompt}. cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
    "Manga": "manga style {prompt}. vibrant, high-energy, detailed, iconic, Japanese comic style",
}
DEFAULT_STYLE_NAME = "3D Model"
STYLE_NAMES = list(STYLES.keys())

MAX_SEED = 1000000000
DEFAULT_SKETCH_GUIDANCE = 0.28
