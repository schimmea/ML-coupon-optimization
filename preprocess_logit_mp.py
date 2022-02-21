import numpy as np
import pandas as pd
import multiprocessing as mp
import sys
from tqdm import tqdm

"""
Multi core data preprocessing for logit model.
"""
def preprocess_logit(baskets_coupons):
    # We always consider 1 shopper at a time
    # Input is a tuple of the basket history and the coupon history for that shopper
    baskets = baskets_coupons[0]
    coupons = baskets_coupons[1]
    shopper = baskets.shopper.values[0]
    
    # Sanity check that both frames actually contain the same shopper
    assert shopper == coupons.shopper.values[0]
    
    # Densify baskets and coupons
    df = pd.DataFrame({"week": sorted(list(range(90))*250),
                       "shopper": [shopper]*90*250,
                       "product": list(range(250))*90})
    
    df["week"] = df["week"].astype("uint8")
    df["shopper"] = df["shopper"].astype("int32")
    df["product"] = df["product"].astype("uint8")
    
    merged = df.merge(baskets, how="left", on=["week", "shopper", "product"]).fillna(0)
    merged["bought"] = merged["bought"].astype("uint8")
    
    # Precalculate the grouping because we use it a lot in the following
    grouped = merged.groupby(['shopper', 'product']).bought
    
    # Calculate the past frequencies first. Need to be careful to not create a leak, hence we shift by 1.
    freqs = grouped.apply(lambda x: x.shift(1).expanding().mean()).fillna(0)
    merged['freqs'] = freqs.astype("float32")
    
    # Same with the past purchase counts
    purchases_last_5_weeks = grouped.apply(lambda x: x.shift(1).fillna(0).rolling(5, min_periods=0).sum())
    merged['purchases_last_5_weeks'] = purchases_last_5_weeks.astype("uint8")

    purchases_last_15_weeks = grouped.apply(lambda x: x.shift(1).fillna(0).rolling(15, min_periods=0).sum())
    merged['purchases_last_15_weeks'] = purchases_last_15_weeks.astype("uint8")

    purchases_last_25_weeks = grouped.apply(lambda x: x.shift(1).fillna(0).rolling(25, min_periods=0).sum())
    merged['purchases_last_25_weeks'] = purchases_last_25_weeks.astype("uint8")
    
    # Finally, add the discounts as last feature
    merged = merged.merge(coupons, how="left").fillna(0)
    merged["discount"] = merged["discount"].astype("uint8")
    return merged
    

if __name__ == '__main__':
    
    # Command line arguments
    if len(sys.argv) > 2:
        cpu_count = int(sys.argv[2])
    else:
        cpu_count = 2
    
    x_random_shoppers = int(sys.argv[1])
    
    # Draw random shopper indexes
    shoppers_ix = np.random.choice(100000, size=x_random_shoppers, replace=False)
    
    print(f"\nCPU Count: {cpu_count}")
    print(f"Number of shoppers being processed: {x_random_shoppers}")
    
    # Load basket data, drop the price and perform some data type transformations to save space
    print("Loading basket data...")
    baskets = pd.read_parquet("data/baskets.parquet")
    baskets.drop(columns="price", inplace=True)
    baskets["week"] = baskets["week"].astype("uint8")
    baskets["shopper"] = baskets["shopper"].astype("int32")
    baskets["product"] = baskets["product"].astype("uint8")
    baskets["bought"] = 1
    baskets["bought"] = baskets["bought"].astype("uint8")
    
    # Here we split the dataframe by shopper and put the separate subframes into a list
    # The idea is that the feature engineering jobs don't overlap for different shoppers, so we can split them up and distribute to multiple cores
    print("Splitting basket data...")
    split_baskets = [y for x, y in baskets[baskets.shopper.isin(shoppers_ix)].groupby('shopper', as_index=False)]
    del baskets
    
    # We do the exact same for coupons
    print("Loading coupon data...")
    coupons = pd.read_parquet("data/coupons.parquet")
    coupons["week"] = coupons["week"].astype("uint8")
    coupons["shopper"] = coupons["shopper"].astype("int32")
    coupons["product"] = coupons["product"].astype("uint8")
    coupons["discount"] = coupons["discount"].astype("uint8")
    
    print("Splitting coupon data...")
    split_coupons = [y for x, y in coupons[coupons.shopper.isin(shoppers_ix)].groupby('shopper', as_index=False)]
    del coupons
    print("Preprocessing data...")
    
    logit_out = []
    
    # Create a pool of multiple clustered worker threads
    p = mp.Pool(cpu_count)
    # ... and a progress bar
    pbar = tqdm(total=x_random_shoppers)
    
    try:
        # This is the main loop where the basket and coupon frames are distributed to the workers
        for x in p.imap_unordered(preprocess_logit, zip(split_baskets, split_coupons), chunksize=1):
            logit_out.append(x)
            pbar.update(1)
        pbar.close()
        p.close()
        p.join()
    except KeyboardInterrupt:
        pbar.close()
        p.close()
        p.join()
        p.terminate()
    
    # Afterwards, we concatenate the result into a single dataframe and sort it, since the pooled workload was returned unsorted
    print("Concatenating and sorting...")
    logit_out = pd.concat(logit_out).sort_values(["week", "shopper", "product"])
    
    # Lastly, save the frame
    print("Saving data...")
    logit_out.to_parquet(f"data/logitprep_random_{x_random_shoppers}.parquet")
