"""Ground-truth NSRTs for the LIS thesis scrub environment."""

from typing import Dict, Set

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, ParameterizedOption, Predicate, Type


class LISThesisScrubGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the LIS thesis scrub environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:  # pragma: no cover
        return {"lis_thesis_scrub"}

    @staticmethod
    def get_nsrts(
        env_name: str, types: Dict[str, Type], predicates: Dict[str,
                                                                Predicate],
        options: Dict[str,
                      ParameterizedOption]) -> Set[NSRT]:  # pragma: no cover
        # NSRTs will be learned via predicate invention
        return set()
