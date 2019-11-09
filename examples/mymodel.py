from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertLayer, BertPooler, BertEncoder, BertConfig
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch

class BertForSequenceClassificationNq(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceClassificationNq, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.top_layer = BertLayer(config)
        self.att_on_bert = None
        if getattr(config, 'att_on_bert', None):
            # self.att_on_bert = BertLayer(config)
            self.att_num_layers = config.att_num_layers
            self.att_on_bert_config = BertConfig(num_hidden_layers=self.att_num_layers)
            self.att_on_bert = BertEncoder(self.att_on_bert_config)
            self.pooler = BertPooler(config)
            self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        else:
            self.classifier = nn.Linear(config.hidden_size * 2, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None,
                input_ids_a=None, token_type_ids_a=None, attention_mask_a=None,
                input_ids_b=None, token_type_ids_b=None, attention_mask_b=None):

        # outputs_original = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
        #                     attention_mask=attention_mask, head_mask=head_mask)
        # pooled_output = self.dropout(pooled_output)
        outputs_a = self.bert(input_ids_a, position_ids=None, token_type_ids=token_type_ids_a, attention_mask=attention_mask_a)
        outputs_b = self.bert(input_ids_b, position_ids=None, token_type_ids=token_type_ids_b, attention_mask=attention_mask_b)
        if self.att_on_bert:
            sequence_outputs_a = outputs_a[0]
            sequence_outputs_b = outputs_b[0]
            sequence_outputs_b = sequence_outputs_b[:,1:,:]
            sequence_outputs = torch.cat((sequence_outputs_a, sequence_outputs_b), dim=1)
            attention_mask_b = attention_mask_b[:, 1:]
            attention_mask = torch.cat((attention_mask_a, attention_mask_b), dim=1)
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            head_mask = [None] * self.att_num_layers
            sequence_outputs_att = self.att_on_bert(sequence_outputs, attention_mask=extended_attention_mask, head_mask=head_mask)
            pooled_output = self.pooler(sequence_outputs_att[0])

        else:
            pooled_output_a = outputs_a[1]
            pooled_output_a = self.dropout(pooled_output_a)
            pooled_output_b = outputs_b[1]
            pooled_output_b = self.dropout(pooled_output_b)
            pooled_output = torch.cat((pooled_output_a, pooled_output_b), dim=1)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs_a[2:] + outputs_b[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)