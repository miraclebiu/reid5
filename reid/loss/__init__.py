
from __future__ import absolute_import

from .xentropy_sac import XentropyLoss_SAC
from .quadruplet_sac import Quadruplet_SAC
from .triplet_sac import Triplet_SAC
from .triplet_weighted_sac import Triplet_Weighted_SAC
from .mgn_loss import MGN_loss
from .rank_loss import Rank_loss
from .loss_container import *
# __all__ = [
#     'XentropyLoss_biu',
#     'Triplet_biu',
#     'Triplet_weighted_biu',
#     'Quadruplet_biu',
#     'MGN_loss'
# ]
support_loss = [XentropyLoss_SAC, Quadruplet_SAC,
                Triplet_SAC, Triplet_Weighted_SAC, 
                MGN_loss, Rank_loss ]

