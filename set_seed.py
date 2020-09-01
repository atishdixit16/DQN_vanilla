def set_seed(seed_value):
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    from tensorflow import set_random_seed
    set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    from keras import backend as K
    import tensorflow as tf
    # for later versions:
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    tf.keras.backend.set_session(sess)

    import torch
    torch.manual_seed(seed_value)