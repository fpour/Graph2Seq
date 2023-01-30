# Graph2Seq

This repository contains the code to replicate the results in **Graph2Seq** paper.

##### Resources:
* [Graph2Seq: Graph to Sequence Learning with Attention-based Neural Networks](https://arxiv.org/abs/1804.00823).
* Official Github repository: [here](https://github.com/IBM/Graph2Seq).
* Github repository of the dataset: [WikiSQL](https://github.com/salesforce/WikiSQL).

In addition to the main resources, I also checked the following references:
* [PyTorch Seq2Seq Tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
  * The corresponding data can be downloaded from [here](https://download.pytorch.org/tutorial/data.zip).
* [PyG_gcn](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py)
  * The corresponding data can be downloaded from [here](https://github.com/kimiyoung/planetoid/tree/master/data).
* [PyG_MP_Net](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html)


### Requirements
* `python==3.9` was used for the implementation.
* Other dependencies:
```{bash}
torch==1.13
torch_geometric==2.2
numpy==1.23
pandas==1.5
```
* Please refer to `requirements.txt` for all the dependencies.


## Project Structure
The Graph2Seq model contains two main components. 
In what follows, an overview of the files implementing these component (and their current state) is elaborated.

**Graph2Seq:**
* Graph Encoder & Graph Embedding
    * ✅`graph_encoder.py`: 
    Two different variation of a GNN model (_GCN_ & _Bi-GCN_) are implemented.
    _Bi-GCN_ follows the GNN architecture explained in the paper. 
    The underlying convolution layer used in _Bi-GCN_ is implemented in `conv_layer.py`.
    The graph encoder is complete and its functionality can be tested separately (by running `graph_encoder.py` file).
* Attention-Based Decoder
  * ✅ `attention_decoder.py`:
  This file contains the implementation of the attention-based decoder. 
  To check its correct functionality, it has been tested in `Seq2Seq_model.py` as the decoder part of a sequence-to-sequence translation task.

Other files and their functionalities are as follows:
* ✅`params.py`: contains different parameters.
* ✅`parser.py`: parses the required arguments.
* ✅`utils.py`: contains some utility functions and classes.
* 👀`main.py`: controls the main flow of the procedure that consists of:
  * Data loading and processing: 
  The data should be loaded and processed to the correct format usable by the model in this part of the code.
  This part is incomplete 👀. However, I wrote the assumptions about the data format, 
  which also specifies what steps I need to take to prepare the data.
  * Model definition:
  Here, different components of the model, their corresponding optimizers, and the criterion are defined.
  * Training & validation:
  Here, the training and validation takes place. 
  The training and validation procedure is implemented in `train.py` file.
  * Testing: 
  Here the trained model is tested with the test split of the data. 
  The evaluation of the test split is implemented in `eval.py` file.
* ✅`train.py`: This file contains the training and validation procedure.
* ✅`eval.py`: This file contains the evaluation procedure.  
* 👀`data_proc/data_loading.py`: 
  Here, the data should be loaded and processed to the correct format.
  The original data contains natural language question, SQL queries, and SQL tables.
  The SQL queries need to be converted to graph so that they can be used by the graph encoder.
















