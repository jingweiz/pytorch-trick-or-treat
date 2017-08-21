* `requires_grad` VS `volatile`
    * > Volatile differs from requires_grad in how the flag propagates. If there’s even a single volatile input to an operation, its output is also going to be volatile. Volatility spreads accross the graph much easier than non-requiring gradient - you only need a single volatile leaf to have a volatile output, while you need all leaves to not require gradient to have an output the doesn’t require gradient. Using volatile flag you don’t need to change any settings of your model parameters to use it for inference. It’s enough to create a volatile input, and this will ensure that no intermediate states are saved.
    * `Variable` newly created (by default):
        ```py
        requires_grad = False
        volatile = False
        ```
    * `Variable` output from any `nn.Module` (by default):
        ```py
        requires_grad = True
        volatile = False
        ```

* `vb = vb.detach()` VS `vb = Variable(vb.data)`
    * after `vb = vb.detach()`:
        ```py
        requires_grad = False
        volatile = Unchanged
        ```
    * after `vb = Variable(vb.data)`:
        ```py
        requires_grad = False
        volatile = False
        ```
