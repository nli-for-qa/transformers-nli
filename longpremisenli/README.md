We reuse the template provided by the transformers package for our experiemental runs of RoBERTa. Relevant files can be found in the examples directory.

- `run_qa_as_nli.py` is used for training
- `run_qa_as_nli_eval.py` is used for ad-hoc evaluatuations of trained models


The other relevant repositories are:
- https://github.com/nli-for-qa/conversion for the conversion rules
- https://github.com/nli-for-qa/transformers-bart for the BART model used for neural conversion
- https://github.com/nli-for-qa/utils notebooks for dataset creation and evaluations
