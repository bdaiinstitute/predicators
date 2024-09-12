from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Sequence, Set

import PIL.Image

from predicators.pretrained_model_interface import OpenAIVLM
from predicators.settings import CFG
from predicators.spot_utils.perception.perception_structs import \
    RGBDImageWithContext
from predicators.structs import Object, State, VLMGroundAtom, VLMPredicate
from predicators.utils import get_object_combinations

###############################################################################
#                      VLM Predicate Evaluation Related                       #
###############################################################################

# Initialize VLM
available_choices = [
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-mini"
]
vlm = OpenAIVLM(model_name=available_choices[2], detail="auto")


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
    
Some predicates may include 'KnownAsTrue' or 'KnownAsFalse'.
You should already respond 'Yes' or 'No' but never 'Unknown'.
If you don't know answer for predicates with 'KnownAsTrue' or 'KnownAsFalse', say 'No'.
    
Here are VLM predicates we have, note that they are defined over typed variables.
Example: (<predicate-name> <obj1-variable>:<obj1-type> ...)
VLM Predicates (separated by line or newline character):
{vlm_predicates}

Examples (separated by line or newline character):
Do these predicates hold in the following images?
1. Inside(apple:object, bowl:container)
2. On(apple:object, table:surface)
3. Blocking(apple:object, orange:object)
4. Blocking(apple:object, apple:object)
5. On(apple:object, apple:object)
6. On(apple:object, bowl:container)
7. EmptyKnownTrue(bowl:container)
8. EmptyKnownFalse(bowl:container)
9. Inside(bowl:container, bowl:container)

Answer (in a single word Yes/No for each question):
1. Yes
2. No
3. Yes
4. No
5. No
6. Yes
7. Yes
8. No
9. No

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

    logging.info(f"VLM predicate evaluation for: \n{question}")
    logging.debug(f"Prompt: {full_prompt}")

    # TODO update the logic here for retrying
    vlm_responses = vlm.sample_completions(
        prompt=full_prompt,
        imgs=images,
        temperature=0.2,
        seed=int(time.time()),
        num_completions=1,
    )
    logging.debug(f"VLM response 0: {vlm_responses[0]}")

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


def vlm_predicate_batch_query(
    queries: List[str],
    images: Dict[str, RGBDImageWithContext],
    predicate_prompts: Optional[List[str]] = None,
) -> List[bool]:
    """Use queries generated from VLM predicates to evaluate them via VLM in
    batch.

    The VLM takes a list of queries and images in current observation to
    evaluate them.
    """

    # Assemble the full prompt
    numbered_queries = [f"{i+1}. {query}" for i, query in enumerate(queries)]
    question = '\n'.join(numbered_queries)
    vlm_predicates = '\n'.join(predicate_prompts) if predicate_prompts else ''
    full_prompt = vlm_predicate_batch_eval_prompt.format(
        vlm_predicates=vlm_predicates, question=question)

    image_list = [
        PIL.Image.fromarray(v.rotated_rgb) for _, v in images.items()
    ]

    logging.info(f"VLM predicate evaluation input (with prompt): \n{question}")
    if CFG.vlm_eval_verbose:
        logging.info(f"Prompt: {full_prompt}")
    else:
        logging.debug(f"Prompt: {full_prompt}")        

    while True:
        vlm_responses = vlm.sample_completions(
            prompt=full_prompt,
            imgs=image_list,
            temperature=0.1,
            seed=int(time.time()),
            num_completions=1,
        )
        logging.info(f"VLM response 0: {vlm_responses[0]}")

        # Parse the responses
        responses = vlm_responses[0].strip().split('\n')
        if len(responses) != len(queries):
            logging.warning(f"[Warning] Number of responses ({len(responses)}) does not match number of queries ({len(queries)}). Retrying...")
            print(f"VLM responses: {vlm_responses[0]}")
            continue

        results = []
        retry = False
        for i, r in enumerate(responses):
            if any(x in r for x in ['Yes', 'No']):
                if 'Yes' in r:
                    results.append(True)
                elif 'No' in r:
                    results.append(False)
            else:
                logging.warning(f"Invalid response in line {i}: {r}. Retrying...")
                retry = True
                break

        if not retry:
            # If no invalid responses, break the while loop
            break

    return results


def vlm_predicate_batch_classify(
        atoms: Set[VLMGroundAtom],
        images: Dict[str, RGBDImageWithContext],
        predicates: Optional[Set[VLMPredicate]] = None,
        get_dict: bool = True
) -> Dict[VLMGroundAtom, bool] | Set[VLMGroundAtom]:
    """Use VLM to evaluate a set of atoms in a given state."""
    # Get the queries for the atoms
    queries = [atom.get_query_str() for atom in atoms]
    if predicates is not None:
        predicate_prompts = [p.pddl_str() for p in predicates]
    else:
        predicate_prompts = None

    if len(queries) == 0:
        return {}

    queries_print = [
        atom.get_query_str(include_prompt=False) for atom in atoms
    ]
    logging.info(f"VLM predicate evaluation queries: {queries_print}")

    # Call VLM to evaluate the queries
    results = vlm_predicate_batch_query(queries, images, predicate_prompts)

    # Update the atoms with the results
    if get_dict:
        # Return all ground atoms with True/False/None
        return {atom: result for atom, result in zip(atoms, results)}
    else:
        # Only return True ground atoms
        return {atom for atom, result in zip(atoms, results) if result}


def get_vlm_atom_combinations(objects: Sequence[Object],
                              preds: Set[VLMPredicate]) -> Set[VLMGroundAtom]:
    atoms = set()
    for pred in preds:
        for choice in get_object_combinations(objects, pred.types):
            atoms.add(VLMGroundAtom(pred, choice))
    return atoms
