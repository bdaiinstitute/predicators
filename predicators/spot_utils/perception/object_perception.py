from __future__ import annotations

import logging
import time
from typing import Dict, List, Set, Sequence

import PIL.Image

from predicators.pretrained_model_interface import OpenAIVLM
from predicators.spot_utils.perception.perception_structs import RGBDImageWithContext
from predicators.structs import State, VLMGroundAtom, Object, VLMPredicate
from predicators.utils import get_object_combinations

###############################################################################
#                      VLM Predicate Evaluation Related                       #
###############################################################################

# Initialize VLM
vlm = OpenAIVLM(model_name="gpt-4-turbo", detail="auto")

# Engineer the prompt for VLM
vlm_predicate_eval_prompt = """
Your goal is to answer questions related to object relationships in the 
given image(s) from the cameras of a Spot robot.
We will use following predicate-style descriptions to ask questions:
    Inside(object1, container)
    Blocking(object1, object2)
    On(object, surface)

Examples:
Does this predicate hold in the following image?
Inside(apple, bowl)
Answer (in a single word): Yes/No

Actual question:
Does this predicate hold in the following image?
{question}
Answer (in a single word):
"""

vlm_predicate_batch_eval_prompt = """
Your goal is to answer questions related to object relationships in the 
given image(s) from the cameras of a Spot robot. Each question is independent 
while all questions rely on the same set of Spot images at a certain moment.
We will use following predicate-style descriptions to ask questions:
    Inside(object1, container)
    Blocking(object1, object2)
    On(object, surface)

Examples (separated by line or newline character):
Do these predicates hold in the following images?
Inside(apple:object, bowl:container)
On(apple:object, table:surface)
Blocking(apple:object, orange:object)
Blocking(apple:object, apple:object)
On(apple:object, apple:object)

Answer (in a single word Yes/No/Unknown for each question, unknown if can't tell from given images):
Yes
No
Unknown
No
No

Actual questions (separated by line or newline character):
Do these predicates hold in the following images?
{question}

Answer (in a single word Yes/No for each question):
"""

# Provide some visual examples when needed
vlm_predicate_eval_prompt_example = ""


def vlm_predicate_classify(question: str, state: State) -> bool | None:
    """Use VLM to evaluate (classify) a predicate in a given state.

    TODO: Next, try include visual hints via segmentation ("Set of Masks")
    """
    full_prompt = vlm_predicate_eval_prompt.format(question=question)
    images_dict: Dict[str, RGBDImageWithContext] = state.camera_images
    images = [
        PIL.Image.fromarray(v.rotated_rgb) for _, v in images_dict.items()
    ]

    logging.info(f"VLM predicate evaluation for: {question}")
    logging.info(f"Prompt: {full_prompt}")

    vlm_responses = vlm.sample_completions(
        prompt=full_prompt,
        imgs=images,
        temperature=0.2,
        seed=int(time.time()),
        num_completions=1,
    )
    logging.info(f"VLM response 0: {vlm_responses[0]}")

    vlm_response = vlm_responses[0].strip().lower()
    if vlm_response == "yes":
        return True
    elif vlm_response == "no":
        return False
    elif vlm_response == "unknown":
        return None
    else:
        logging.error(
            f"VLM response not understood: {vlm_response}. Treat as None.")
        return None


def vlm_predicate_batch_query(queries: List[str], images: Dict[str, RGBDImageWithContext]) -> List[bool]:
    """Use queries generated from VLM predicates to evaluate them via VLM in batch.

    The VLM takes a list of queries and images in current observation to evaluate them.
    """

    # Assemble the full prompt
    question = '\n'.join(queries)
    full_prompt = vlm_predicate_batch_eval_prompt.format(question=question)

    image_list = [
        PIL.Image.fromarray(v.rotated_rgb) for _, v in images.items()
    ]

    logging.info(f"VLM predicate evaluation for: {question}")
    logging.info(f"Prompt: {full_prompt}")

    vlm_responses = vlm.sample_completions(
        prompt=full_prompt,
        imgs=image_list,
        temperature=0.2,
        seed=int(time.time()),
        num_completions=1,
    )
    logging.info(f"VLM response 0: {vlm_responses[0]}")

    # Parse the responses
    responses = vlm_responses[0].strip().lower().split('\n')
    results = []
    for i, r in enumerate(responses):
        assert r in ['yes', 'no', 'unknown'], f"Invalid response in line {i}: {r}"
        if r == 'yes':
            results.append(True)
        elif r == 'no':
            results.append(False)
        else:
            results.append(None)
    assert len(results) == len(queries), "Number of responses should match queries."

    return results


def vlm_predicate_batch_classify(atoms: Set[VLMGroundAtom], images: Dict[str, RGBDImageWithContext],
                                 get_dict: bool = True) -> Dict[VLMGroundAtom, bool] | Set[VLMGroundAtom]:
    """Use VLM to evaluate a set of atoms in a given state."""
    # Get the queries for the atoms
    queries = [atom.get_query_str() for atom in atoms]

    if len(queries) == 0:
        return {}

    logging.info(f"VLM predicate evaluation queries: {queries}")

    # Call VLM to evaluate the queries
    results = vlm_predicate_batch_query(queries, images)

    # Update the atoms with the results
    if get_dict:
        return {atom: result for atom, result in zip(atoms, results)}
    else:
        hold_atoms = set()
        for atom, result in zip(atoms, results):
            if result:
                hold_atoms.add(atom)
        return hold_atoms


def get_vlm_atom_combinations(objects: Sequence[Object], preds: Set[VLMPredicate]) -> Set[VLMGroundAtom]:
    atoms = set()
    for pred in preds:
        for choice in get_object_combinations(objects, pred.types):
            atoms.add(VLMGroundAtom(pred, choice))
    return atoms
