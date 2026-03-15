# ================================
# Model Architecture
# ================================

embedding_dim = 256

hidden_dim = 512

num_layers = 2

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

batch_size = 128

learning_rate = 3e-4

num_epochs = 30


# ================================
# Dataset / Vocabulary
# ================================

max_vocab_size = 20000


# ================================
# Teacher Forcing
# ================================

teacher_forcing_ratio = 0.5