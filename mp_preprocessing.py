import pandas as pd
import numpy as np
import multiprocessing as mp
import pickle
import sys
from tqdm import tqdm

"""
Multi core data preprocessing for main model. 
"""
def make_history_and_coupons(df_tuple):
    # We always consider 1 shopper at a time
    # Input is a tuple of the basket history and the coupon history for that shopper
    baskets_df = df_tuple[0]
    coupons_df = df_tuple[1]
    
    # Do a cross product first to minimize intermediate join results
    df = baskets_df.merge(coupons_df, how="outer").fillna(0)
    
    shopper = baskets_df.shopper.values[0]
    # Sanity check that both subframes actually contain the same shopper
    assert shopper == coupons_df.shopper.values[0], "Shoppers in baskets and coupons are not the same!!"
    
    num_weeks = 90
    num_products = 250
    # Densify the data:
    tmp = pd.DataFrame({"week": sorted(list(range(num_weeks))*num_products),
                        "shopper": [shopper]*num_weeks*num_products,
                        "product": list(range(num_products))*num_weeks})
    
    # Merge the dense frame with the cross product from earlier to fill it
    merged = tmp.merge(df, how="left", on=["week", "shopper", "product"]).fillna(0).drop(columns="shopper")
    
    # Now construct a pivot table with the products in the rows and the weeks in the columns
    # It has a 2nd level column index of "bought" and "discount"
    baskets_coupons_out = merged.pivot_table(index="product", columns="week")
    
    # The "bought" superindex encodes the purchase events as {0, 1}
    baskets_out = baskets_coupons_out["bought"].values.astype("uint8")
    
    # The "discount" superindex gives the discount for all products at any given week
    coupons_out = baskets_coupons_out["discount"].values.astype("uint8")
    
    return baskets_out, coupons_out
    

if __name__ == '__main__':
    
    # Command line arguments
    if len(sys.argv) > 2:
        cpu_count = int(sys.argv[2])
    else:
        cpu_count = 2
    
    first_x_shoppers = int(sys.argv[1])
    
    print(f"\nCPU Count: {cpu_count}")
    print(f"Number of shoppers being processed: {first_x_shoppers}")
    
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
    split_baskets = [y for x, y in baskets[baskets.shopper < first_x_shoppers].groupby('shopper', as_index=False)]
    del baskets
    
    # We do the exact same for coupons
    print("Loading coupon data...")
    coupons = pd.read_parquet("data/coupons.parquet")
    coupons["week"] = coupons["week"].astype("uint8")
    coupons["shopper"] = coupons["shopper"].astype("int32")
    coupons["product"] = coupons["product"].astype("uint8")
    coupons["discount"] = coupons["discount"].astype("uint8")
    
    print("Splitting coupon data...")
    split_coupons = [y for x, y in coupons[coupons.shopper < first_x_shoppers].groupby('shopper', as_index=False)]
    del coupons

    print("Preprocessing data...")
    
    hist_out = np.empty((first_x_shoppers, 250, 90), dtype="uint8")
    coupon_out = np.empty((first_x_shoppers, 250, 90), dtype="uint8")
    
    # Create a pool of multiple clustered worker threads
    p = mp.Pool(cpu_count)
    # ... and a progress bar
    pbar = tqdm(total=first_x_shoppers)
    # Initialize the count that represents the current shopper
    count = 0
    
    try:
        # This is the main loop where the basket and coupon frames are distributed to the workers
        for x in p.imap(make_history_and_coupons, zip(split_baskets, split_coupons), chunksize=1):
            hist_out[count] = x[0]
            coupon_out[count] = x[1]
            pbar.update(1)
            count += 1
        pbar.close()
        p.close()
        p.join()
    except KeyboardInterrupt:
        pbar.close()
        p.close()
        p.join()
        p.terminate()
    
    # Finally save the data
    print("Pickling data...")
    with open(f'data/histories_first_{first_x_shoppers}.pkl', 'wb') as file_name:
        pickle.dump(hist_out, file_name)
    with open(f'data/coupons_first_{first_x_shoppers}.pkl', 'wb') as file_name:
        pickle.dump(coupon_out, file_name)
