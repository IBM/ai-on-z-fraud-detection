## Scope

This repository provides TensorFlow source code for building and training credit card fraud models using an LSTM and a GRU.

## Summary
The models included in this repository are multi-layer LSTM or GRU models that analyze time series data to predict whether a credit card transaction is fraudulent.

The models consist of a recurrent neural network (RNN) with 2 layers of long short-term memory (LSTM) or gated recurrent unit (GRU), 200 units in each layer, followed by a dense layer.  There is one output, which is Fraud/Non-fraud.  A sequence of 7 transactions is used as the input to model. 

In a transactional environment such as z/OS CICS, new data is generated from card terminal transactions.  The transaction is received by a z/OS application that then must process it.  As part of this processing, validation occurs.  This can include the use of AI to provide insight on whether the transaction is potentially fraudulent.

The TensorFlow saved models were converted to the ONNX format using the tf2onnx 1.10.0 library and are also included in this repo.  The models were converted using the following template:
```
import tf2onnx
spec = (tf.TensorSpec((7, 16, 220), tf.float32, name="input"),)
output_path = new_model.name + ".onnx"
onnx_model = tf2onnx.convert.from_keras(new_model, spec, output_path=output_path)
```
## Environment:
- s390x
- Python 3.11.6
- pandas==1.5.3
- numpy==1.24.3
- scikit-learn==1.2.2
- sklearn-pandas==2.2.0
- tensorflow==2.12.1
- tf2onnx=1.16.1

## The dataset used in this repo can be found here:  
https://github.com/IBM/TabFormer/tree/main/data/credit_card

## License

If you would like to see the detailed LICENSE click [here](LICENSE).

