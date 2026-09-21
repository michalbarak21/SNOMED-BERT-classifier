# one BERT encoder, two linear heads on top - one for the SNOMED code and one
# for presence. both trained together so the encoder learns both.

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer


# short names, so you can write --base clinicalbert instead of the whole path. 
# anything not here goes to huggingface as is.
BASE_MODELS = {
    "bert": "bert-base-uncased",
    "bert-cased": "bert-base-cased",	
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "biobert": "dmis-lab/biobert-base-cased-v1.2",
    "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "sapbert": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
}


def resolve_base_model(name):
    if name in BASE_MODELS:
        return BASE_MODELS[name]
    return name 

def load_tokenizer(base_model):
    return AutoTokenizer.from_pretrained(resolve_base_model(base_model))


class DualHeadClassifier(nn.Module):
    """bert encoder + two heads (codes and presence)"""

    def __init__(self, base_model, num_codes, num_presences, dropout=0.1, pooling="cls"):
        super().__init__()
        self.base_model = resolve_base_model(base_model)
        self.pooling =  pooling 
        config = AutoConfig.from_pretrained(self.base_model)
        self.encoder = AutoModel.from_pretrained(self.base_model, config = config)
        hidden = config.hidden_size
        # dont move these 3 lines around! torch draws the random init in the
        # order the layers are built  so you get a diffrent model with  same seed..
        self.dropout = nn.Dropout(dropout)
        self.code_head = nn.Linear(hidden , num_codes)
        self.presence_head = nn.Linear(hidden, num_presences)

    def _pool(self, hs, attention_mask):
        # cls = just the first token
        if self.pooling == "cls":
            return hs[:, 0]
        # else average the real tokens, skip the paddnig 
        mask = attention_mask.unsqueeze(-1).to(hs.dtype)
        s = (hs * mask).sum(1)
        cnt = mask.sum(1).clamp(min=1e-9)  # clamp or an all-pad row divides by zero
        return s/ cnt

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        # not every encoder wants token_type_ids so only pass it if we got one
        if token_type_ids is not None: 
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        pooled = self._pool(out.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        return self.code_head(pooled), self.presence_head(pooled)	


def parameter_summary(model):
    """how many params we actually train"""
    tr = 0
    tot =  0
    for p in model.parameters():
        n = p.numel()
        tot += n
        if p.requires_grad: 
            tr+= n
    pct = 100*tr / tot
    return f"{tr:,} trainable / {tot:,} total ({pct:.2f}%)"
