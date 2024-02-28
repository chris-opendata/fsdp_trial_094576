from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration


class Model(nn.Module):
    def __init__(self, args, logger, **kwargs):
        super().__init__()

        # Instantiate baseline module
        logger.info("Creating seq2seq model from pretrained weights.")
        self.seq2seq = BartForConditionalGeneration.from_pretrained(
                            args.base_model_pretrained_name,
                            cache_dir=args.pretrained_model_cache_dir)

        vocab_size = kwargs.pop("vocab_size")
        self.seq2seq.resize_token_embeddings(vocab_size)

        h_dim = kwargs.pop("h_dim")
        s_dim = kwargs.pop("s_dim")
        self.bl = nn.Parameter(torch.FloatTensor(1, h_dim, s_dim))

    def forward(self, batch):
        outputs = self.seq2seq(
                        **batch,
                        output_attentions=True,
                        output_hidden_states=True,
                    )

        m_output = {}
        m_output["cost"] = outputs.loss.cpu()

        h_x = outputs.encoder_hidden_states[-1]
        h_y = outputs.decoder_hidden_states[-1]

        # Some extended example code
        x = h_x @ self.bl.to(device=h_x.device)
        x = x @ h_y.transpose(-1,-2)

        return m_output

    @torch.no_grad()
    def generate(
        self,
        batch,
        options,
        **model_kwargs,
    ):
        inputs = batch[0]

        return self.seq2seq.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **model_kwargs
                )
