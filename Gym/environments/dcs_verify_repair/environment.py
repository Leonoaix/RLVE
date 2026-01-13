import random
from typing import Optional, List, Dict, Tuple
from Gym.environment import VerifiableEnvironment


class DCS_VerifyRepair_Environment(VerifiableEnvironment):
    """
    Given constraints + an initial candidate assignment x_init:
    (1) identify violated constraints, (2) propose a repaired assignment x'.

    Output (EXACTLY two lines inside <answer>):
      Line1: violated indices (0-based) separated by spaces, OR 'none'
      Line2: repaired x vector (ints separated by spaces)

    Reward in [0,1]:
      total = w_v * vio_score + w_i * imp_score
      vio_score = map F1(pred, gold) to [0,1]
      imp_score = map improvement to [0,1]
      improvement = max(0, sat(x') - sat(x_init)) / max(1, M - sat(x_init))
    """

    prompt_template = r"""There are {N} integers x[0], x[1], ..., x[{N_minus_1}]. They should satisfy the following {M} inequations:
{inequations}

An initial candidate assignment is:
{candidate}

Task:
1) Identify which inequations are violated by the candidate (by index).
2) Provide a repaired assignment that satisfies more inequations than the candidate.

You MUST answer using the following format and NOTHING ELSE.
Inside <answer>...</answer>, output EXACTLY two lines and no explanation.

<answer>
(indices in ascending order separated by spaces, OR none)
x0 x1 ... x{N_minus_1}
</answer>
"""

    def __init__(
        self,
        num_range: int = 5,
        candidate_perturb: int = 3,
        wrong_format: float = 0.0,
        weight_violation: float = 0.3,
        weight_improve: float = 0.7,
        passing_reward_threshold: float = 0.4,
        **kwargs
    ):
        if "answer_markers" not in kwargs or kwargs["answer_markers"] is None:
            kwargs["answer_markers"] = ("<answer>", "</answer>")
        super().__init__(**kwargs)

        assert 0.0 <= weight_violation <= 1.0 and 0.0 <= weight_improve <= 1.0
        assert abs(weight_violation + weight_improve - 1.0) < 1e-6, "weights must sum to 1"

        self.number_range = num_range
        self.candidate_perturb = candidate_perturb
        self.rewards = dict(
            wrong_format=wrong_format,
            w_v=weight_violation,
            w_i=weight_improve,
        )
        self.passing_reward_threshold = float(passing_reward_threshold)

    @staticmethod
    def _f1(pred: set, gold: set) -> float:
        if not pred and not gold:
            return 1.0
        if not pred or not gold:
            return 0.0
        tp = len(pred & gold)
        prec = tp / len(pred) if pred else 0.0
        rec = tp / len(gold) if gold else 0.0
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)

    @staticmethod
    def _count_satisfied(x: List[int], inequations: List[Tuple[int, int]], results: List[int]) -> int:
        return sum(int(x[i] - x[j] <= rhs) for (i, j), rhs in zip(inequations, results))

    @staticmethod
    def _violations(x: List[int], inequations: List[Tuple[int, int]], results: List[int]) -> List[int]:
        bad = []
        for idx, ((i, j), rhs) in enumerate(zip(inequations, results)):
            if not (x[i] - x[j] <= rhs):
                bad.append(idx)
        return bad

    def _generate(self) -> None:
        assert "N" in self.parameter and "M" in self.parameter
        N = self.parameter["N"]
        M = self.parameter["M"]
        assert N >= 2 and M >= 1

        x_star = [random.randint(-N, +N) for _ in range(N)]
        inequations = random.sample(
            [(i, j) for i in range(N) for j in range(N) if i != j],
            min(M, N * (N - 1))
        )
        results = [x_star[i] - x_star[j] + random.randint(0, self.number_range) for i, j in inequations]

        x_init = x_star[:]
        for k in range(N):
            x_init[k] += random.randint(-self.candidate_perturb, self.candidate_perturb)

        gold_v = self._violations(x_init, inequations, results)
        base_sat = self._count_satisfied(x_init, inequations, results)

        self.parameter.update(dict(
            inequations=inequations,
            results=results,
            x_init=x_init,
            gold_violations=gold_v,
            base_satisfied=base_sat,
        ))

    def _prompt_generate(self) -> str:
        N = self.parameter["N"]
        lines = []
        for idx, (((i, j), rhs)) in enumerate(zip(self.parameter["inequations"], self.parameter["results"])):
            lines.append(f"({idx}) x[{i}] - x[{j}] <= {rhs}")
        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            M=len(lines),
            inequations="\n".join(lines),
            candidate=" ".join(map(str, self.parameter["x_init"])),
        )

    def _process(self, answer: Optional[str]) -> Optional[Dict]:
        if answer is None:
            return None
        lines = [ln.strip() for ln in answer.strip().splitlines() if ln.strip() != ""]
        if len(lines) != 2:
            return None

        # line 1: violations
        vio_line = lines[0].lower().strip()
        if vio_line == "none":
            vio = []
        else:
            try:
                vio_line = vio_line.replace(",", " ")
                parts = vio_line.split()
                cap = max(64, 2 * self.parameter.get("M", 16))
                if len(parts) > cap:
                    parts = parts[:cap]
                vio = [int(p) for p in parts]
            except Exception:
                return None

        # line 2: repaired x (strip parentheses + commas)
        try:
            line2 = lines[1].replace("(", " ").replace(")", " ").replace(",", " ")
            x_new = list(map(int, line2.split()))
        except Exception:
            return None

        return dict(vio=vio, x=x_new)

    def scorer(self, output: str) -> float:
        parsed = self.processor(output)
        if parsed is None:
            return self.rewards["wrong_format"]

        vio = parsed["vio"]
        x_new = parsed["x"]

        N = self.parameter["N"]
        inequations = self.parameter["inequations"]
        results = self.parameter["results"]
        M = len(inequations)

        # allow extra numbers, but require at least N
        if len(x_new) < N:
            return self.rewards["wrong_format"]
        if len(x_new) > N:
            x_new = x_new[:N]

        # filter OOR indices, then canonicalize
        vio = [idx for idx in vio if 0 <= idx < M]
        vio = sorted(set(vio))

        gold = set(self.parameter["gold_violations"])
        pred = set(vio)
        f1 = self._f1(pred, gold)           # [0,1]

        base_sat = self.parameter["base_satisfied"]
        new_sat = self._count_satisfied(x_new, inequations, results)
        delta = new_sat - base_sat
        if base_sat >= M:
            imp_score = 1.0
        else:
            denom = max(1, M - base_sat)
            imp_score = delta / denom 
            imp_score = max(0.0, min(1.0, imp_score))

        total = self.rewards["w_v"] * f1 + self.rewards["w_i"] * imp_score
        return float(max(0.0, min(1.0, total)))
