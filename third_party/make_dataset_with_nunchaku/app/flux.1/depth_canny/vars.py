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
DEFAULT_STYLE_NAME = "None"
STYLE_NAMES = list(STYLES.keys())

MAX_SEED = 1000000000
DEFAULT_INFERENCE_STEP_CANNY = 50
DEFAULT_GUIDANCE_CANNY = 30.0

DEFAULT_INFERENCE_STEP_DEPTH = 30
DEFAULT_GUIDANCE_DEPTH = 10.0

HEIGHT = 1024
WIDTH = 1024

EXAMPLES = {
    "canny": [
        [
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png",
            "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts.",
            DEFAULT_STYLE_NAME,
            STYLES[DEFAULT_STYLE_NAME],
            50,
            30,
            0,
        ],
        [
            "https://huggingface.co/mit-han-lab/svdq-int4-flux.1-fill-dev/resolve/main/example.png",
            "A wooden basked of several individual cartons of strawberries.",
            DEFAULT_STYLE_NAME,
            STYLES[DEFAULT_STYLE_NAME],
            50,
            30,
            1,
        ],
    ],
    "depth": [
        [
            "https://huggingface.co/mit-han-lab/svdq-int4-flux.1-canny-dev/resolve/main/logo_example.png",
            "A logo of 'MIT HAN Lab'.",
            "Fantasy art",
            STYLES["Fantasy art"],
            30,
            10,
            2,
        ],
        [
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png",
            "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts.",
            DEFAULT_STYLE_NAME,
            STYLES[DEFAULT_STYLE_NAME],
            30,
            10,
            0,
        ],
        [
            "https://huggingface.co/mit-han-lab/svdq-int4-flux.1-fill-dev/resolve/main/example.png",
            "A wooden basket of several individual cartons of strawberries.",
            DEFAULT_STYLE_NAME,
            STYLES[DEFAULT_STYLE_NAME],
            30,
            10,
            1,
        ],
    ],
}
