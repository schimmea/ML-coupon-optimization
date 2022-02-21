# Product and customer-specific coupon assignment using Machine Learning (Winter 2020/21)

In order to increase per-coupon revenue, targeted allocation of coupons based on individual customer preferences has proven to be fairly effective in real world settings (Gabel & Timoshenko, 2020).

In this work, we attempt to tackle the problem of product and customer-specific coupon assignment by implementing
a multi-part neural network architecture first proposed by Gabel and Timoshenko (2020). This model aims to detect
product level purchase probabilities for certain given coupon allocations, enabling the selection of those coupons which
maximize uplift (Reimers & Xie, 2019). To achieve this, it processes both customer-specific purchase histories and
coupon assignments as well as latent product-related information. Our goal is to use this model to select optimal,
revenue maximizing coupon allocations for 2,000 customers in our dataset.
