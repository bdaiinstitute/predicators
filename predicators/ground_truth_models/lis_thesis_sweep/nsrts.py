"""Ground-truth NSRTs for the LIS thesis sweep environment."""

from typing import Dict, Set

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, ParameterizedOption, Predicate, Type


class LISThesisSweepGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the LIS thesis sweep environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:  # pragma: no cover
        return {"lis_thesis_sweep"}

    @staticmethod
    def get_nsrts(
        env_name: str, types: Dict[str, Type], predicates: Dict[str,
                                                                Predicate],
        options: Dict[str,
                      ParameterizedOption]) -> Set[NSRT]:  # pragma: no cover
        # NSRTs will be learned via predicate invention
        return set()
