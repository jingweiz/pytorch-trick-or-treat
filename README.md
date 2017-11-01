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

* inherited classes of `nn.Module`:
    * all submodules must be either `nn.` or `nn.ModuleList`, otherwise they will not be handled correctly. e.g. the following example will give errors since `self.list` is not put onto `cuda` by `self.type(torch.cuda.FloatTensor)`. Thus, `torch.save(model.state_dict(), "...")` and `model.load_state_dict(torch.load("..."))` will also not work correctly for `model.list`
        ```py
        import torch
        import torch.nn as nn
        from torch.autograd import Variable

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.layer = nn.Linear(10, 10)
                self.modulelist = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])
                self.list = [nn.Linear(10, 10) for i in range(10)]

                self.type(torch.cuda.FloatTensor)

            def forward(self, input_vb):
                x_vb = self.layer(input_vb)
                print("passed through layer")
                for i in range(len(self.modulelist)):
                    x_vb = self.modulelist[i](x_vb)
                print("passed through modulelist")
                for i in range(len(self.list)):
                    x_vb = self.list[i](x_vb)
                print("passed through list")
                return x_vb

        model = Model()
        input_vb = Variable(torch.randn(3, 10))
        output_vb = model(input_vb.cuda())
        ```
