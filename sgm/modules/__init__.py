from .encoders.modules import (GeneralConditioner,
                               GeneralConditionerWithControl,
                               PreparedConditioner)

UNCONDITIONAL_CONFIG = {
    "target": "sgm.modules.GeneralConditioner",
    "params": {"emb_models": []},
}
