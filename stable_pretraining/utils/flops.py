from torch.utils.flop_counter import FlopCounterMode
from torch.utils._python_dispatch import TorchDispatchMode
from contextlib import contextmanager


class FLOPBudgetExceeded(Exception):
    """Exception raised when FLOP budget is exceeded."""

    def __init__(self, budget: int, current: int, operation: str = ""):
        self.budget = budget
        self.current = current
        self.operation = operation
        super().__init__(
            f"FLOP budget exceeded: {current:,} FLOPs used (budget: {budget:,})"
            + (f" at operation: {operation}" if operation else "")
        )


class BudgetedFlopCounterMode(TorchDispatchMode):
    """FLOP counter with budget enforcement using composition.

    Wraps FlopCounterMode for counting, adds budget checking.
    """

    def __init__(self, flop_counter: FlopCounterMode, budget: int):
        self._flop_counter = flop_counter
        self.budget = budget
        self._last_op = ""

    @property
    def total_flops(self) -> int:
        return self._flop_counter.get_total_flops()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self._last_op = getattr(func, "__name__", str(func))

        # Call the function (FlopCounterMode will intercept and count)
        result = func(*args, **(kwargs or {}))

        # Check budget after FlopCounterMode has counted
        current = self.total_flops
        if current > self.budget:
            raise FLOPBudgetExceeded(self.budget, current, self._last_op)

        return result


@contextmanager
def flop_budget(budget: int, display: bool = False):
    """Context manager that counts FLOPs and raises when budget is exceeded.

    :param budget: Maximum number of FLOPs allowed
    :param display: Whether to print FLOP breakdown on exit
    :yields: Counter object with .total_flops property

    Example:
        try:
            with flop_budget(1e9) as counter:
                for i in range(1000):
                    out = model(x)
                    print(f"FLOPs so far: {counter.total_flops:,}")
        except FLOPBudgetExceeded as e:
            print(f"Stopped at {e.current:,} FLOPs")
    """
    budget = int(budget)

    # FlopCounterMode does the counting
    flop_counter = FlopCounterMode(display=display)

    # Our mode checks the budget after each op
    budget_checker = BudgetedFlopCounterMode(flop_counter, budget)

    # Stack: budget_checker (top) -> flop_counter (bottom) -> actual ops
    # When op is called: budget_checker intercepts -> calls func ->
    # flop_counter intercepts and counts -> actual op runs ->
    # returns to budget_checker which checks budget
    with flop_counter:
        with budget_checker:
            yield budget_checker
