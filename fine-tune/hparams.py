# import tensorflow as tf
#from text import symbols
import json
ENV_PATH="../config/msp.json"
with open(ENV_PATH, 'r') as f:
    env = json.load(f)
MEL_PATH = env["MEL_PATH"]

def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self

    # hparams = tf.contrib.training.HParams(
    hparams = AttrDict(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=50,
        iters_per_checkpoint=100,
        seed=1234,
        dynamic_loss_scaling=True,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,

        ################################
        # Data Parameters              #
        ################################
        training_list='../data/esd_list_0011/training_mel_list.txt',
        validation_list='../data/esd_list_0011/evaluation_mel_list.txt',
        mel_mean_std=MEL_PATH+'/norm/ESD/mel_mean_std.npy',
        # training_list='data/list/training_mel_list.txt',
        # validation_list='data/list/evaluation_mel_list.txt',
        # mel_mean_std='data/mel_spec/norm/mel_mean_std.npy',
        ################################
        # Data Parameters              #
        ################################
        n_mel_channels=80,
        n_spc_channels=1025,
        n_symbols=41, #
        pretrain_n_speakers=99, #

        n_speakers=5, #
        predict_spectrogram=False,

        ################################
        # Model Parameters             #
        ################################

        symbols_embedding_dim=512,

        # Text Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,
        text_encoder_dropout=0.5,

        # Audio Encoder parameters
        spemb_input=False,
        n_frames_per_step_encoder=2,  
        audio_encoder_hidden_dim=512,
        AE_attention_dim=128,
        AE_attention_location_n_filters=32,
        AE_attention_location_kernel_size=51,
        beam_width=10,

        # hidden activation 
        # relu linear tanh
        hidden_activation='tanh',

        #Speaker Encoder parameters
        speaker_encoder_hidden_dim=256,
        speaker_encoder_dropout=0.2,
        speaker_embedding_dim=128,


        #Speaker Classifier parameters
        SC_hidden_dim=512,
        SC_n_convolutions=3,
        SC_kernel_size=1,

        # Decoder parameters
        feed_back_last=True,
        n_frames_per_step_decoder=2,
        decoder_rnn_dim=512,
        prenet_dim=[256,256],
        max_decoder_steps=1000,
        stop_threshold=0.5,
    
        # Attention parameters
        attention_rnn_dim=512,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=17,

        # PostNet parameters
        postnet_n_convolutions=5,
        postnet_dim=512,
        postnet_kernel_size=5,
        postnet_dropout=0.5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        #weight_decay=1e-6,
        weight_decay=1e-4,
        grad_clip_thresh=5.0,
        batch_size=32,
        warmup = 7,
        decay_rate = 0.5,
        decay_every = 7,


        contrastive_loss_w=30.0,
        speaker_encoder_loss_w=1.0,
        text_classifier_loss_w=1.0,
        speaker_adversial_loss_w=20.,
        speaker_classifier_loss_w=0.1,
        ce_loss=False
    )

    return hparams



