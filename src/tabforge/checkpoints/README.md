# TabFORGE best model weights

The official TabFORGE v1 reusable backbone is hosted in the
[`XiangjianJiang/TabFORGE`](https://huggingface.co/XiangjianJiang/TabFORGE)
model repository. It is downloaded to the Hugging Face cache when fitting uses
`checkpoint="pretrained"`; the checkpoint bytes remain excluded from TabFORGE
source and wheel distributions.

The canonical `checkpoint.pt` contains only learned Transformer parameters for
the decoder and diffusion components. It intentionally omits
the TabPFN encoder, token identities, observation-mask parameters, output
heads/detokenizers, normalization statistics, reference embeddings, fitted
preprocessors, and training data.

Use `checkpoint=None` for a fully fresh initialization or pass a canonical
checkpoint path for transfer.
Architecture metadata lives inside the checkpoint. After fitting or
fine-tuning, use `save_checkpoint()` for exact inference restoration; its
default policy adds the reference bank for generators and omits it for
prediction, embedding, and imputation estimators.
