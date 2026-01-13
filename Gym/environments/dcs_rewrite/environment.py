import random
import re
from typing import Optional, List, Tuple
from Gym.environment import VerifiableEnvironment


class DCS_Rewrite_Environment(VerifiableEnvironment):
    """
    Rewrite each inequation x[i] - x[j] <= c into x[i] <= x[j] + c.

    Output: lines matching pattern; junk lines are ignored (training-friendly).
    Reward: (correct/M)^beta mapped to [0,1]
    """

    prompt_template = r"""There are {N} integers x[0], x[1], ..., x[{N_minus_1}]. They satisfy the following {M} inequations:
{inequations}

Task:
Rewrite EACH inequation into the unified form:
x[i] <= x[j] + c

You MUST answer using the following format and NOTHING ELSE.
Inside <answer>...</answer>, output ONLY rewritten lines. Do NOT include any explanation.

<answer>
x[i] <= x[j] + c
...
</answer>
"""

    _rewrite_pat = re.compile(r"^x\[(\d+)\]\s*<=\s*x\[(\d+)\]\s*\+\s*(-?\d+)\s*$")

    def __init__(
        self,
        num_range: int = 5,
        wrong_format: float = 0,
        rewarding_beta: float = 2.0,
        passing_reward_threshold: float = 0.5,
        **kwargs
    ):
        if "answer_markers" not in kwargs or kwargs["answer_markers"] is None:
            kwargs["answer_markers"] = ("<answer>", "</answer>")
        super().__init__(**kwargs)

        self.number_range = num_range
        self.rewards = dict(
            wrong_format=wrong_format,
            rewarding_beta=rewarding_beta,
        )
        self.passing_reward_threshold = float(passing_reward_threshold)

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

        self.parameter["inequations"] = inequations
        self.parameter["results"] = results

    def _prompt_generate(self) -> str:
        N = self.parameter["N"]
        lines = [f"x[{i}] - x[{j}] <= {rhs}" for (i, j), rhs in zip(self.parameter["inequations"], self.parameter["results"])]
        return self.prompt_template.format(N=N, N_minus_1=N - 1, M=len(lines), inequations="\n".join(lines))

    def _process(self, answer: Optional[str]) -> Optional[List[Tuple[int, int, int]]]:
        if answer is None:
            return None
        lines = [ln.strip() for ln in answer.strip().splitlines() if ln.strip() != ""]
        parsed: List[Tuple[int, int, int]] = []

        M = len(self.parameter.get("inequations", []))
        max_keep = max(32, 2 * M) if M > 0 else 32

        for ln in lines:
            m = self._rewrite_pat.match(ln)
            if not m:
                continue
            parsed.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
            if len(parsed) >= max_keep:
                break
        return parsed


    def scorer(self, output: str) -> float:
        parsed = self.processor(output)
        if parsed is None or len(parsed) == 0:
            return self.rewards["wrong_format"]

        inequations = self.parameter["inequations"]
        results = self.parameter["results"]
        M = len(inequations)

        gold = set((i, j, rhs) for (i, j), rhs in zip(inequations, results))
        pred = set(parsed)
        correct = len(gold & pred)

        frac = correct / M
        return frac ** self.rewards["rewarding_beta"]
