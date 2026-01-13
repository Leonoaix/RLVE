from typing import Dict, List, Optional
from Gym.parameter_controller import ParameterController


class DCS_Rewrite_ParameterController(ParameterController):
    """
    Rewrite is output-length sensitive (≈ M lines).
    So we cap M to keep generation fast and stable.
    """

    def __init__(
        self,
        M_multiple_list: Optional[List[float]] = None,
        N_init: int = 3,
        M_cap: int = 8,          # <<< 新增：硬上限（建议 6~10）
        N_cap: int = 12,         # <<< 可选：防止 N 无限长导致 prompt 变长
        **kwargs
    ):
        super().__init__(**kwargs)
        self.N = N_init
        self.M_multiple = M_multiple_list if M_multiple_list is not None else [1.0, 1.3, 1.6, 2.0]
        self.M_cap = int(M_cap)
        self.N_cap = int(N_cap)

    def update(self) -> None:
        self.N = int(self.N * 1.1 + 1)
        if self.N > self.N_cap:
            self.N = self.N_cap

    def get_parameter_list(self) -> List[Dict]:
        params = []
        for m in self.M_multiple:
            M = max(1, int(m * self.N))
            M = min(M, self.M_cap)    # <<< 关键：cap M
            params.append(dict(N=self.N, M=M))
        return params
