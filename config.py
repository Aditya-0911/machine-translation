# ================================
# Model Architecture
# ================================

embedding_dim = 128

hidden_dim = 256

num_layers = 1

dropout = 0.1


# ================================
# Special Tokens
# ================================

pad_idx = 0
sos_idx = 1
eos_idx = 2


# ================================
# Training Hyperparameters
# ================================

batch_size = 64

learning_rate = 3e-4

num_epochs = 5


# ================================
# Dataset / Vocabulary
# ================================

max_vocab_size = 20000


# ================================
# Teacher Forcing
# ================================

teacher_forcing_ratio = 0.5