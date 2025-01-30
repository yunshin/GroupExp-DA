from .anchor_head_template import AnchorHeadTemplate
from .anchor_head_single import AnchorHeadSingle
from .point_intra_part_head import PointIntraPartOffsetHead
from .point_head_simple import PointHeadSimple
from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_single_ours_exp_self2 import AnchorHeadSingle_Ours_EXP_Self2
from .anchor_head_single_ours_exp_1rpn import AnchorHeadSingle_Ours_EXP_1RPN
from .anchor_head_single_ours_exp_0rpn import AnchorHeadSingle_Ours_EXP_0RPN
from .anchor_head_single_ours_exp_0_8rpn import AnchorHeadSingle_Ours_EXP_0_8RPN
from .anchor_head_template_ours import AnchorHeadTemplate_Ours
from .anchor_head_single_ours_exp_0_8rpn_one import AnchorHeadSingle_Ours_EXP_0_8RPN_One
from .anchor_head_single_ours_exp_0_8rpn_one_ori import AnchorHeadSingle_Ours_EXP_0_8RPN_One_Ori
__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointHeadSimple': PointHeadSimple,
    'AnchorHeadMulti': AnchorHeadMulti,



    'AnchorHeadSingle_Ours_EXP_Self2': AnchorHeadSingle_Ours_EXP_Self2,
    'AnchorHeadSingle_Ours_EXP_1RPN': AnchorHeadSingle_Ours_EXP_1RPN,
    'AnchorHeadSingle_Ours_EXP_0RPN': AnchorHeadSingle_Ours_EXP_0RPN,
    'AnchorHeadSingle_Ours_EXP_0_8RPN': AnchorHeadSingle_Ours_EXP_0_8RPN,
    'AnchorHeadSingle_Ours_EXP_0_8RPN_One': AnchorHeadSingle_Ours_EXP_0_8RPN_One,
    'AnchorHeadSingle_Ours_EXP_0_8RPN_One_Ori': AnchorHeadSingle_Ours_EXP_0_8RPN_One_Ori,
    'AnchorHeadTemplate_Ours': AnchorHeadTemplate_Ours
}
