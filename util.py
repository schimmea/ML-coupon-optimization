import numpy as np
import pandas as pd
from main_model import *
from elasticity_funcs import *
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import average_precision_score

# The main validation function, tailored to the main model
def iterate_epochs(histories, 
                   coupons,
                   model=None,
                   weeks_to_validate=10,
                   validate_every=30,
                   shoppers_for_scoring=10000, 
                   J=250,
                   H=40, 
                   L=50, 
                   max_epochs=30,
                   min_improve=0.001,
                   patience=5,
                   batch_size=32,
                   learning_rate=0.001,
                   calculate_elasticities=False,
                   verbose=1):
    
    # Inputs:
    #   histories:              Preprocessed histories (not diced)
    #   coupons:                Preprocessed coupons (not diced)
    #   model:                  Only if an existing model should be trained further, otherwise create a new one
    #   weeks_to_validate:      Size of hold-out set
    #   validate_every:         Week intervals for calulation of validation metrics
    #   shoppers_for_scoring:   Number of random shoppers to use for scoring
    #   J:                      Number of products
    #   H:                      Number of convolutional filters
    #   L:                      Size of bottleneck
    #   max_epochs:             Maximum epochs to train the model for
    #   min_improve:            Minimum improvement to not trigger patience
    #   patience:               Early stopping after triggering patience this often
    #   batch_size:             Batch size for model training
    #   learning_rate:          Adam learning rate
    #   calculate_elasticities: Whether to calculate elasticities at validation
    #   verbose:                Generate output? Does not affect validation metric output.
    
    # Outputs:
    #   The trained model
    
    
    # Get random indices for scoring
    shopper_ixs = np.random.choice(histories.shape[0], size=shoppers_for_scoring, replace=False)
    
    # Initialize patience trigger count
    patience_used = 0
    
    # First week to start training at
    train_start = 0
    # Last training week
    train_end = histories.shape[2] - days_to_validate - 1

    # Make inputs depending on number of validation weeks 
    histories_x_train = histories[:, :, train_start:train_end]
    coupons_x_train = coupons[:, :, train_end].reshape((coupons.shape[0], 250, 1))
    y_train = histories[:, :, train_end].reshape((histories.shape[0], 250, 1))
    
    if verbose:
        print(f"\nTraining on weeks {train_start} to {train_end - 1} with labels of week {train_end}...\n")
    
    # If no model is provided, a new one is created (see main_model.py)
    if model is None:
        model = build_model(J, train_end, H, L, learning_rate=learning_rate)

    model.compile(loss=BinaryCrossentropy(), optimizer=Adam(learning_rate), metrics=[AUC()])
    
    # Used for patience trigger
    loss_history = []
    best_loss = np.inf
    
    for epoch in range(max_epochs):
        print(f"Epoch {epoch+1}")
        
        # Train the model for 1 epoch
        model.fit([histories_x_train, coupons_x_train], y_train, epochs=1, verbose=1, batch_size=batch_size, validation_split=0)
        
        log_losses = []
        aucs = []
        aps = []
        owns = []
        withins = []
        cross_pos = []
        cross_neg = []
        cross_zero = []
        
        # Validation metrics loop
        if (epoch + 1)%validate_every == 0:
            for week in range(days_to_validate):
                
                # Construct validation input
                val_start = train_start + week + 1
                val_end = train_end + week + 1
                histories_x_val = histories[shopper_ixs, :, val_start:val_end]
                coupons_x_val = coupons[shopper_ixs, :, val_end].reshape((shoppers_for_scoring, 250, 1))
                y_val = histories[shopper_ixs, :, val_end].reshape((shoppers_for_scoring, 250, 1))

                # Get predictions
                preds = model.predict([histories_x_val, coupons_x_val])
                
                weekly_log_losses = []
                weekly_aucs = []
                weekly_aps = []
                
                # Calculate metrics over all shoppers and average
                for shopper in range(shoppers_for_scoring):
                    weekly_log_losses.append(log_loss(y_val[shopper], preds[shopper], labels=[1, 0]))
                    # Sometimes a shopper doesn't buy in a week so we avoid errors here
                    if(sum(y_val[shopper]) > 0):
                        weekly_aucs.append(roc_auc_score(y_val[shopper], preds[shopper], labels=[1, 0]))
                        weekly_aps.append(average_precision_score(y_val[shopper], preds[shopper]))
                    else:
                        weekly_aucs.append(np.NaN)
                        weekly_aps.append(np.NaN)
                
                log_losses.append(np.mean(weekly_log_losses))
                aucs.append(np.nanmean(weekly_aucs))
                aps.append(np.nanmean(weekly_aps))
                
                if calculate_elasticities:
                    # Calculate elasticities (see elasticity_funcs.py)
                    elasticity_matrix = get_elasticities(model, histories_x_val, num_shoppers=int(shoppers_for_scoring/10), progress_bar=False)
                    owns.append(own_product_elasticity(elasticity_matrix))
                    cats = sorted(list(range(25))*10)  # Derived from P2V; hardcoded for convenience
                    withins.append(within_category_elasticity(elasticity_matrix, cats))
                    crosses = cross_category_elasticity(elasticity_matrix, cats, thresh=0.05)
                    cross_pos.append(crosses[0])
                    cross_neg.append(crosses[1])
                    cross_zero.append(crosses[2])
                else:
                    owns.append(np.NaN)
                    withins.append(np.NaN)
                    cross_pos.append(np.NaN)
                    cross_neg.append(np.NaN)
                    cross_zero.append(np.NaN)
                
                # Print weekly scores
                if week==0:
                    print("Week\tLog Loss\tAUC\tAP Score\tOwn-Product\tWithin-Cat\tCross-Pos\tCross-Neg\tCross-Zero")
                print(f"{val_end}\t{log_losses[-1]:.3f}\t\t{aucs[-1]:.3f}\t{aps[-1]:.3f}\t\t{owns[-1]:.3f}\t\t{withins[-1]:.3f}\t\t{cross_pos[-1]:.3f}\t\t{cross_neg[-1]:.3f}\t\t{cross_zero[-1]:.3f}")
            
            loss_history.append(np.mean(log_losses))
            
            # Print total scores
            print(f"\nTotal\t{loss_history[-1]:.3f}\t\t{np.nanmean(aucs):.3f}\t{np.nanmean(aps):.3f}\t\t{np.mean(owns):.3f}\t\t{np.mean(withins):.3f}\t\t{np.mean(cross_pos):.3f}\t\t{np.mean(cross_neg):.3f}\t\t{np.mean(cross_zero):.3f}\n")
            
            # Patience trigger stuff
            if best_loss - loss_history[-1] > min_improve:
                best_loss = loss_history[-1]
                patience_used = 0
            else:
                patience_used += 1
            
            if patience_used == patience:
                print(f"\nEarly stopping. Best epoch number: {epoch + 1 - patience}")
                return model
        
    return model


def optimize_coupons(shopper, model, histories, verbose=True):
    # This function implements the greedy coupon optimization
    # Inputs:
    #   shopper:    Shopper to consider
    #   model:      Trained model
    #   histories:  Preprocessed histories (Diced! Has to fit the model.)
    
    # Output:
    #   (250 x 1) optimal coupon vector
    
    coupons = np.zeros((250, 1))
    discounts = [15, 20, 25, 30]
    
    # This price index is defined in "Main_Model.ipynb". It is basically the max price for every product, which means the unreduced price.
    price_index = pd.read_parquet("data/price_index.parquet")
    
    max_profit = 0
    
    # Only consider the current shopper
    histories_input = histories[shopper]
    
    if verbose:
            # Here we predict the revenue without any coupons as a baseline
            probs = model.predict([np.tile(histories_input, [1, 1, 1]),
                                   np.tile(coupons, [1, 1, 1])])
            revenue = (price_index.price.values.reshape((250, 1)) * probs).sum(1)[0][0]
            print(f"[No coupons]\nRevenue:\t{revenue:.2f}\n")
            last_highest_profit = revenue
            
    # We want 5 coupons
    for c in range(5):
        allocations = []
        # For every product, we try every discount size
        for p in range(250):
            # If a product already has a discount, it must be the optimal one from the last outer loop, so we leave it there
            if coupons[p] != 0:
                continue
            for d in discounts:
                tmp = coupons.copy()
                tmp[p] = d
                # We basically collect a big allocations list
                allocations.append(tmp)
        # Stack the allocations list and calculate the purchase probabilities for every single one
        coupons_input = np.stack(allocations)
        
        # The model fortunately makes this pretty convenient
        probs = model.predict([np.tile(histories_input, [coupons_input.shape[0], 1, 1]),
                               coupons_input])
        # Calculate revenue for every allocation
        revenue = (price_index.price.values.reshape((250, 1)) * (1 - coupons_input/100) * probs).sum(1)
        # The allocation with the highest revenue is chosen
        new_max_profit = np.max(revenue)
        # We don't want negative revenue changes!
        assert new_max_profit > max_profit, f"[Shopper {shopper}] New coupon has effected revenue negatively!"
        max_profit = new_max_profit
        best_alloc = np.where(revenue == max_profit)[0].tolist()[0]
        coupons = coupons_input[best_alloc]
        if verbose:
            difference = max_profit - last_highest_profit
            print(f"""[Best coupon {c+1}]\nProducts:\t{', '.join(np.where(coupons>0)[0].astype('str').tolist())}\nDiscounts:\t{', '.join(coupons[coupons>0].astype('str').tolist())}\nRevenue:\t{max_profit:.2f}\nDifference:\t{"+" if difference > 0 else ""}{difference:.2f}\n""")
            last_highest_profit = max_profit
    return coupons


def optimize_coupons_by_cat(shopper, model, histories, categories, verbose=True):
    # This function implements the greedy coupon optimization but only gives one coupon per category
    # Inputs:
    #   shopper:    Shopper to consider
    #   model:      Trained model
    #   histories:  Preprocessed histories (Diced! Has to fit the model.)
    
    # Output:
    #   (250 x 1) optimal coupon vector
    
    categories = np.array(categories)
    coupons = np.zeros((250, 1))
    discounts = [15, 20, 25, 30]
    
    # This price index is defined in "Main_Model.ipynb". It is basically the max price for every product, which means the unreduced price.
    price_index = pd.read_parquet("data/price_index.parquet")
    
    max_profit = 0
    
    # Only consider the current shopper
    histories_input = histories[shopper]
    
    if verbose:
            # Here we predict the revenue without any coupons as a baseline
            probs = model.predict([np.tile(histories_input, [1, 1, 1]),
                                   np.tile(coupons, [1, 1, 1])])
            revenue = (price_index.price.values.reshape((250, 1)) * probs).sum(1)[0][0]
            print(f"[No coupons]\nRevenue:\t{revenue:.2f}\n")
            last_highest_profit = revenue
    
    visited_cats = []
    # We want 5 coupons
    for c in range(5):
        allocations = []
        # For every product, we try every discount size
        for p in range(250):
            # We don't want to give 2 coupons in a category
            prodcat = categories[p]
            if prodcat in visited_cats:
                continue
            # If a product already has a discount, it must be the optimal one from the last outer loop, so we leave it there
            if coupons[p] != 0:
                continue
            for d in discounts:
                tmp = coupons.copy()
                tmp[p] = d
                # We basically collect a big allocations list
                allocations.append(tmp)
        # Stack the allocations list and calculate the purchase probabilities for every single one
        coupons_input = np.stack(allocations)
        
        # The model fortunately makes this pretty convenient
        probs = model.predict([np.tile(histories_input, [coupons_input.shape[0], 1, 1]),
                               coupons_input])
        # Calculate revenue for every allocation
        revenue = (price_index.price.values.reshape((250, 1)) * (1 - coupons_input/100) * probs).sum(1)
        # The allocation with the highest revenue is chosen
        new_max_profit = np.max(revenue)
        # We don't want negative revenue changes!
        assert new_max_profit > max_profit, f"[Shopper {shopper}] New coupon has effected revenue negatively!"
        max_profit = new_max_profit
        best_alloc = np.where(revenue == max_profit)[0].tolist()[0]
        coupons = coupons_input[best_alloc]
        # Add the product categories from optimal allocation to the visited categories
        visited_cats = categories[np.where(coupons>0)[0]]
        assert len(visited_cats) == len(np.unique(visited_cats))
        if verbose:
            difference = max_profit - last_highest_profit
            print(f"""[Best coupon {c+1}]\nProducts:\t{', '.join(np.where(coupons>0)[0].astype('str').tolist())}\nDiscounts:\t{', '.join(coupons[coupons>0].astype('str').tolist())}\nRevenue:\t{max_profit:.2f}\nDifference:\t{"+" if difference > 0 else ""}{difference:.2f}\n""")
            last_highest_profit = max_profit
    return coupons