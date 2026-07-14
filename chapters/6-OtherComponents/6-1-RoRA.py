class LoRALinear(nn.Module):
    def __init__(self, base_linear, r):
        """
        base_linear: nn.Linear(d_in, d_in)
        """
        super().__init__()

        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        in_dim = base_linear.in_features
        out_dim = base_linear.out_features

        self.A = nn.Linear(in_dim, r, bias=False)
        self.B = nn.Linear(r, out_dim, bias=False)

        nn.init.zeros_(self.B.weight)  # A初始随机化，B初始为0

    def forward(self, x):
        return self.base(x) + self.B(self.A(x))
