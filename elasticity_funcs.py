import numpy as np
from tqdm.notebook import tqdm

# This file is tailored to the main model and will calculate discount elasticities for it
# It uses a number of random shoppers to do this (2000 by default)


def get_elasticities(model, histories, num_shoppers=2000, progress_bar=True):
    # Input:
    #   Trained model
    #   Preprocessed histories (dice them beforehand if the model takes less than 90 weeks!)
    #   Num_shoppers is the number of random shoppers to consider for elasticity calculation
    #   Progress_bar: Whether a progress bar should be displayed
    
    # Output:
    #   250 x 250 Matrix of elasticities
    
    # Define an empty coupon array as baseline
    coupons = np.zeros((250, 1))
    
    # Choose random shopper indices
    shoppers_ix = np.random.choice(histories.shape[0], size=num_shoppers, replace=False)
    
    # Filter the histories by random indices
    histories = histories[shoppers_ix]
    
    # Now predict the base probs for all considered shoppers and all products
    base_probs = model.predict([histories, np.zeros((num_shoppers, 250, 1))])
    
    # Take the average over shoppers
    base_probs = base_probs.mean(axis=0)
    
    # Initialize elasticity matrix
    elasticity_matrix = np.empty((250, 250))
    
    if progress_bar:
        pbar = tqdm(total=250)
   
    for product in range(250):
        # For each product we set the discount to 30 once
        coupons_input = coupons.copy()
        coupons_input[product] = 30
        # Predict probabilities of all shoppers and all products for this allocation
        probs = model.predict([histories, np.tile(coupons_input, [num_shoppers, 1, 1])])
        # Average over shoppers
        avg_probs = probs.mean(axis=0)
        # Calculate elasticities of the current product 
        elasticities = ((avg_probs - base_probs) / (0.3 * base_probs)).reshape((250,))
        # Insert them into the matrix
        elasticity_matrix[product] = elasticities
        if progress_bar:
            pbar.update(1)
    
    if progress_bar:
        pbar.close()
    
    return elasticity_matrix
    

def own_product_elasticity(elasticities):
    return np.mean(np.diag(elasticities))

    
def category_elasticities(elasticities, categories):
    # This function gives out a (250 x c) matrix where c is the number of unique categories in the argument provided
    # It gives the aggregated effect of each product on each category
    # categories needs to be a vector of length 250 containing a category for each product
    # elasticities is the elasticity matrix from the function at the top (or any arbitrary one, for that matter)
    
    effect_matrix = np.empty((250, len(np.unique(categories))))
    
    # For each product
    for prod in range(250):
        # For each category
        for cat in np.unique(categories):
            # Find the products that are in this category
            cat_ixs = np.where(categories==cat)[0]
            # Extract from the elasticity matrix the row vector of the current product, 
            # with the entries belonging to the products of the current catagory 
            subvector = elasticities[prod, cat_ixs]
            elas_sum = np.sum(subvector)
            elas_count = len(subvector)
            # If we found the own category of the product, we exclude it from the calculation
            if prod in cat_ixs:
                elas_sum -= elasticities[prod, prod]
                elas_count -= 1
            # Enter the average over the effects into the category matrix
            effect_matrix[prod, cat] = elas_sum / elas_count
            
    return effect_matrix
    

def within_category_elasticity(elasticities, categories):
    # Use the above function to get category aggregates
    agg_elas_by_cat = category_elasticities(elasticities, categories)
    prod_means = []
    for prod in range(250):
        # Now just get the category of the current product ...
        cat_ix = categories[prod]
        # ...and find the corresponding entry in the category matrix 
        prod_means.append(agg_elas_by_cat[prod, cat_ix])

    # Do this for every product and return the mean of all results
    return np.mean(prod_means)


def cross_category_elasticity(elasticities, categories, thresh=0.05):
    # Again we calculate the category aggregates
    agg_elas_by_cat = category_elasticities(elasticities, categories)
    # Initialize a 3-column array since we look at positive-, negative-, and zero-effect categories
    prod_means = np.empty((250, 3))
    for prod in range(250):
        # Get the category of the current product...
        cat_ix = categories[prod]
        # ... and temporarily delete it from the product's row in the aggregate matrix
        crosscats = np.delete(agg_elas_by_cat[prod], cat_ix)
        
        # The remaining category effects are now all split and aggregated depending on the threshold
        # The ... if ... else ... statements are to suppress numpy RuntimeWarnings since we sometimes average over an empty slice
        pos_mean = np.nanmean(crosscats[crosscats > thresh]) if (crosscats > thresh).any() else np.NaN
        neg_mean = np.nanmean(crosscats[crosscats < -thresh]) if (crosscats < -thresh).any() else np.NaN
        zero_mean = np.nanmean(crosscats[(crosscats <= thresh) & (crosscats >= -thresh)]) if ((crosscats <= thresh) & (crosscats >= -thresh)).any() else np.NaN

        prod_means[prod] = [pos_mean, neg_mean, zero_mean]
    # Return the means aggregated over products
    return np.nanmean(prod_means, axis=0)