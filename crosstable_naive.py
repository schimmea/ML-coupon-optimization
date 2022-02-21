# This file calculates the crosstable for the naive benchmark
# It is done in parallel because the data is simply too large
import pandas as pd
import numpy as np
import multiprocessing as mp
import sys
from tqdm import tqdm


def outer_join_crosstab(basket_coupon_tuple):
    # We always consider 1 shopper at a time
    # Input is a tuple of the basket history and the coupon history for that shopper
    baskets = basket_coupon_tuple[0]
    coupons = basket_coupon_tuple[1]
    shopper = baskets.shopper.values[0]
    # We do a cross product to get 3 out of 4 cases (coupon & bought, no coupon & bought, coupon & not bought)
    outer_join = coupons.merge(baskets, on=["week", "shopper", "product"], how="outer").fillna(0)
    # Then we generate a crosstable. The entry of (no coupon & not bought) will be empty
    crosstab = pd.crosstab(outer_join["discount"], outer_join["bought"]).values
    # We therefore infer it from the number of weeks and products, as well as the sum of the incomplete crosstable
    crosstab[0, 0] = 90*250 - np.sum(crosstab)
    return crosstab


if __name__ == '__main__':
    
    # Command line arguments
    if len(sys.argv) > 1:
        cpu_count = int(sys.argv[1])
    else:
        cpu_count = 2
    
    print(f"\nCPU Count: {cpu_count}")
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
    # The idea is that the jobs don't overlap for different shoppers, so we can split them up and distribute to multiple cores
    print("Splitting basket data...")
    split_baskets = [y for x, y in baskets.groupby('shopper', as_index=False)]
    del baskets
    
    # We do the exact same for coupons
    print("Loading coupon data...")
    coupons = pd.read_parquet("data/coupons.parquet")
    coupons["discount"] = coupons["discount"].astype("uint8")
    coupons["week"] = coupons["week"].astype("uint8")
    coupons["shopper"] = coupons["shopper"].astype("int32")
    coupons["product"] = coupons["product"].astype("uint8")
    coupons["coupon"] = 1
    coupons["coupon"] = coupons["coupon"].astype("uint8")
    
    print("Splitting coupon data...")
    split_coupons = [y for x, y in coupons.groupby('shopper', as_index=False)]
    del coupons
    
    print("Preprocessing data...")
    # crosstab = np.zeros((2, 2)) # For coupons 1 vs. 0
    crosstab = np.zeros((8, 2)) # For coupon sizes
    count = 0
    
    # Create a pool of multiple clustered worker threads
    p = mp.Pool(cpu_count)
    # ... and a progress bar
    pbar = tqdm(total=100000)
    
    try:
        # This is the main loop where the basket and coupon frames are distributed to the workers
        for x in p.imap_unordered(outer_join_crosstab, zip(split_baskets, split_coupons), chunksize=1):
            pbar.update(1)
            crosstab += x
        pbar.close()
        p.close()
        p.join()
    except KeyboardInterrupt:
        pbar.close()
        p.close()
        p.join()
        p.terminate()
 
    # Save the crosstable as .csv
    print("Saving data...")
    np.savetxt("data/crosstable_by_discount.csv", crosstab, delimiter=",")
    