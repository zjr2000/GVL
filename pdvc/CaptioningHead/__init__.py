from .LSTM import LightCaptioner
from .Puppet import PuppetCaptionModel
from .LSTM_DSA import LSTMDSACaptioner
from .GPT import ClipCaptionModel, MappingType
from .Transformer_DSA import TransformerDSACaptioner

def build_captioner(opt):
    if opt.caption_decoder_type == 'none':
        caption_embed = PuppetCaptionModel(opt)

    elif opt.caption_decoder_type == 'light':
        opt.event_context_dim = None
        opt.clip_context_dim = opt.hidden_dim
        if vars(opt).get('enable_pos_emb_for_captioner', False):
            opt.clip_context_dim = opt.hidden_dim * 2
        caption_embed = LightCaptioner(opt)

    elif opt.caption_decoder_type == 'standard':
        opt.event_context_dim = None
        opt.clip_context_dim = opt.hidden_dim
        caption_embed = LSTMDSACaptioner(opt)

    elif opt.caption_decoder_type == 'transformer':
        opt.event_context_dim = None
        opt.clip_context_dim = opt.hidden_dim
        caption_embed = TransformerDSACaptioner(opt)
    elif opt.caption_decoder_type == 'gpt2':
        model = ClipCaptionModel(prefix_length=opt.prefix_length, clip_length=1, prefix_size=opt.prefix_size,
                                 num_layers=opt.prefix_num_mapping_layer, mapping_type=MappingType.MLP, gpt_model=opt.gpt_model, cache_dir=opt.huggingface_cache_dir)
        caption_embed = model
    else:
        raise ValueError('caption decoder type is invalid')
    return caption_embed

