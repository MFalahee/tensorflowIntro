| Term | Definition |
| ---- | ---------- |
| **Supervised learning** | training a model where we have a list or set of correct responses to use for training |
|**Layer** | a set of neurons with 'weights' governing their output.
|**Dense layer**| the hidden juice of ML. A _layer_ that multiplies the input layers by a _matrix_ (weights) and then adds a number called _bias_ to the result|
|**inputShape** | the shape of the input number going to the dense layers|
|**tensors** | primary data structure used in TensorFlow programs. Commonly seen as scalars, vectors, or matrices that can hold ints, floats, or str|
|**optimizer** | the algorithm that governs updates to the model as it sees data -- many optimizers exist in tf|
|**loss**| fn that tells the model how it's doing on each data batch that it's shown.|
|**batchSize** | how much data do we show on each iteration of training (commonly 32-512)|
|**epochs**| how many times the model will go through the entire dataset|

# best practices for training model

1. shuffle data
   -> Don't learn things purely by how the data was entered
   -> Don't be sensitive to subgroups in the data
2. normalize data (0-1, -1-1, anything bigger gets wacko)

# takeaways: **EXAMPLE 1** -->

1. Formulate the task
   - Regression or classification problem?
   - Supervised or unsupervised learning?
   - Shape of the input data? What should the output look like?
2. Prepare the Data
   - Clean the data and inspect it for patterns
   - Shuffle
   - Normalize
   - Convert to tensors
3. Build, and run the model
   - tf.sequential or tf.model --> tf.layers.\*
   - pick an optimizer for the problem
   - loss fn for the problem
4. Test the model
