from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController


class DCS_VerifyRepair_ParameterController(ParameterController):
    """
    Difficulty by:
      - N grows over time
      - M = multiple * N
    Candidate perturb is set in environment init, not here.
    """

    def __init__(self, M_multiple_list: Optional[List[float]] = None, N_init: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.N = N_init
        self.M_multiple = M_multiple_list if M_multiple_list is not None else [1.0, 1.3, 1.6, 2.0]

    def update(self) -> None:
        self.N = int(self.N * 1.1 + 1)

    def get_parameter_list(self) -> List[Dict]:
        return [dict(N=self.N, M=max(1, int(m * self.N))) for m in self.M_multiple]
