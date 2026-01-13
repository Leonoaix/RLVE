import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class DCS_Propose_Environment(VerifiableEnvironment):
    """
    Propose RELATIVE assignment: output d[1..N-1] where d[i] = x[i] - x[0].
    Environment sets x[0] = 0 and reconstructs x = [0] + d to score satisfaction.

    Output: exactly N-1 integers
    Reward: (satisfied/M)^beta mapped to [0,1]
    """

    prompt_template = r"""There are {N} integers x[0], x[1], ..., x[{N_minus_1}]. They satisfy the following {M} inequations:
{inequations}

Task:
Propose a RELATIVE assignment in terms of differences to x[0].
Output d[1], d[2], ..., d[{N_minus_1}] where d[i] = x[i] - x[0].

Note:
- The environment will set x[0] = 0 and reconstruct x[i] = d[i] for i>=1.
- Do NOT output x[0].

You MUST answer using the following format and NOTHING ELSE.
Inside <answer>...</answer>, output ONLY integers separated by spaces.
Inside <answer>...</answer>, output EXACTLY {N_minus_1} integers.

<answer>
d1 d2 ... d{N_minus_1}
</answer>
"""

    def __init__(
        self,
        num_range: int = 5,
        wrong_format: float = 0,
        rewarding_beta: float = 5.0,
        passing_reward_threshold: float = 0.5,
        **kwargs
    ):
        # avoid double answer_markers passed from rlve_rm
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
        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            M=len(lines),
            inequations="\n".join(lines),
        )

    def _process(self, answer: Optional[str]) -> Optional[List[int]]:
        if answer is None:
            return None
        answer = answer.strip().replace(",", " ")
        try:
            vals = list(map(int, answer.split()))
            return vals
        except Exception:
            return None

    def scorer(self, output: str) -> float:
        d = self.processor(output)
        if d is None:
            return self.rewards["wrong_format"]

        N = self.parameter["N"]
        if len(d) != N - 1:
            return self.rewards["wrong_format"]

        x = [0] + d

        sat = sum(
            int(x[i] - x[j] <= rhs)
            for (i, j), rhs in zip(self.parameter["inequations"], self.parameter["results"])
        )
        M = len(self.parameter["inequations"])
        frac = sat / M
        return frac ** self.rewards["rewarding_beta"]
