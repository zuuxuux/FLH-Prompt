"""Model components including frozen backbone and prompt pools.

Available pool algorithms:
- HedgePool: Pure multiplicative weights (can go extinct)
- FixedSharePool: Fixed Share with per-step mixing (prevents extinction)
- TrueFLHPool: Follow the Leading History with sub-algorithm ensemble (adaptive regret)
- FLHPromptPool: Alias for FixedSharePool (legacy compatibility)
"""

from flh_prompts.models.flh_pool import (
    HedgePool,
    FixedSharePool,
    TrueFLHPool,
    FLHPromptPool,
)
