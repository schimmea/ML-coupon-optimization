# Final assignment Machine Learning in Marketing (Winter 2020/21)

In this project we aim to maximize revenue by allocating personalized coupons to customers. Therefor, we implement a multi-part neural network as proposed by Gabel and Timoshenko (2020).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Please insert the provided data into python/data.

To execute the code and predict the best suited coupons for each customer open and run the Main_Model.ipynb in the python folder.

Trained model weights are in python/saved_weights.

To view the baseline creation, open and run Logistic_Regression.ipynb and Naive_Baseline.ipynb in the python folder.

The saved logistic regression model and crosstable for naive benchmark are in the python/data folder.
Other preprocessed data needs to be created, see instructions in the respective notebooks.

Product2Vec embeddings and the co-occurrence matrix are created in the Product2Vec subfolder, which is forked from https://github.com/sbstn-gbl/p2v-map

Open and run p2vmap.ipynb. The p2v outputs are saved in python/Product2Vec/data.

## Final predictions

The final predictions are located in the root directory as coupon_index.parquet.

