"""
Parameters shared between different modules

Date:
    - Jan. 28, 2023
"""

MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1

RAND_SEED = 42
teacher_forcing_ratio = 0.5


# data-related parameters
num_node_init_feats = 40
num_layer_dec = 1

# output paths
tr_loss_fig_name = "outputs/training_loss.png"
eval_out_attn_fig_name = "outputs/eval_out_attns.png"


# input related
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)
