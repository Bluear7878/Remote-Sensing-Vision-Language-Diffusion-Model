MAX_IMAGE_SIZE = 2048
MAX_SEED = 1000000000

DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024

PROMPT_TEMPLATES = {
    "None": "{prompt}",
    "Anime": "{prompt}, nm22 style",
    "GHIBSKY Illustration": "GHIBSKY style, {prompt}",
    "Realism": "{prompt}",
    "Yarn Art": "{prompt}, yarn art style",
    "Children Sketch": "sketched style, {prompt}",
}

LORA_PATHS = {
    "Anime": {
        "name_or_path": "alvdansen/sonny-anime-fixed",
        "weight_name": "araminta_k_sonnyanime_fluxd_fixed.safetensors",
    },
    "GHIBSKY Illustration": {
        "name_or_path": "aleksa-codes/flux-ghibsky-illustration",
        "weight_name": "lora.safetensors",
    },
    "Realism": {
        "name_or_path": "mit-han-lab/FLUX.1-dev-LoRA-Collections",
        "weight_name": "realism.safetensors",
    },
    "Yarn Art": {
        "name_or_path": "linoyts/yarn_art_Flux_LoRA",
        "weight_name": "pytorch_lora_weights.safetensors",
    },
    "Children Sketch": {
        "name_or_path": "mit-han-lab/FLUX.1-dev-LoRA-Collections",
        "weight_name": "sketch.safetensors",
    },
}

SVDQ_LORA_PATH_FORMAT = "mit-han-lab/svdquant-lora-collection/svdq-int4-flux.1-dev-{name}.safetensors"
SVDQ_LORA_PATHS = {
    "Anime": SVDQ_LORA_PATH_FORMAT.format(name="anime"),
    "GHIBSKY Illustration": SVDQ_LORA_PATH_FORMAT.format(name="ghibsky"),
    "Realism": SVDQ_LORA_PATH_FORMAT.format(name="realism"),
    "Yarn Art": SVDQ_LORA_PATH_FORMAT.format(name="yarn"),
    "Children Sketch": SVDQ_LORA_PATH_FORMAT.format(name="sketch"),
}


EXAMPLES = {
    "schnell": [
        [
            "An elegant, art deco-style cat with sleek, geometric fur patterns reclining next to a polished sign that "
            "reads 'MIT HAN Lab' in bold, stylized typography. The sign, framed in gold and silver, "
            "exudes a sophisticated, 1920s flair, with ambient light casting a warm glow around it.",
            4,
            1,
        ],
        [
            "Pirate ship trapped in a cosmic maelstrom nebula, rendered in cosmic beach whirlpool engine, "
            "volumetric lighting, spectacular, ambient lights, light pollution, cinematic atmosphere, "
            "art nouveau style, illustration art artwork by SenseiJaye, intricate detail.",
            4,
            2,
        ],
        [
            "A worker that looks like a mixture of cow and horse is working hard to type code.",
            4,
            3,
        ],
        [
            "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. "
            "She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. "
            "She wears sunglasses and red lipstick. She walks confidently and casually. "
            "The street is damp and reflective, creating a mirror effect of the colorful lights. "
            "Many pedestrians walk about.",
            4,
            4,
        ],
        [
            "Cozy bedroom with vintage wooden furniture and a large circular window covered in lush green vines, "
            "opening to a misty forest. Soft, ambient lighting highlights the bed with crumpled blankets, a bookshelf, "
            "and a desk. The atmosphere is serene and natural. 8K resolution, highly detailed, photorealistic, "
            "cinematic lighting, ultra-HD.",
            4,
            5,
        ],
        [
            "A photo of a Eurasian lynx in a sunlit forest, with tufted ears and a spotted coat. The lynx should be "
            "sharply focused, gazing into the distance, while the background is softly blurred for depth. Use cinematic "
            "lighting with soft rays filtering through the trees, and capture the scene with a shallow depth of field "
            "for a natural, peaceful atmosphere. 8K resolution, highly detailed, photorealistic, "
            "cinematic lighting, ultra-HD.",
            4,
            6,
        ],
    ],
    "dev": [
        [
            'a cyberpunk cat holding a huge neon sign that says "SVDQuant is lite and fast", wearing fancy goggles and '
            "a black leather jacket.",
            25,
            3.5,
            "None",
            0,
            2,
        ],
        ["a dog wearing a wizard hat", 28, 3.5, "Anime", 1, 23],
        [
            "a fisherman casting a line into a peaceful village lake surrounded by quaint cottages",
            28,
            3.5,
            "GHIBSKY Illustration",
            1,
            233,
        ],
        ["a man in armor with a beard and a sword", 25, 3.5, "Realism", 0.9, 2333],
        ["a panda playing in the snow", 28, 3.5, "Yarn Art", 1, 23333],
        ["A squirrel wearing glasses and reading a tiny book under an oak tree", 24, 3.5, "Children Sketch", 1, 233333],
    ],
}
