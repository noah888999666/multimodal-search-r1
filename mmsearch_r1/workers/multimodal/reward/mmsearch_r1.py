import json

import torch
from verl import DataProto

from mmsearch_r1.utils.reward_score_mm import _default_compute_score


class MMSearchR1_RewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score

    def extract_responses_list(
        self,
        tokenizer,
        input_ids: torch.Tensor,  # User Prompt + All Responses
        multi_turn_response_mask: torch.Tensor,  # 0,0,0,...,1,1,1,...,0,0,0,...,1,1,1
    ) -> list:
        diff = torch.diff(multi_turn_response_mask, prepend=torch.tensor([0], device=multi_turn_response_mask.device))
        starts = torch.where(diff == 1)[0]
        mask_appended = torch.cat(
            [multi_turn_response_mask, torch.tensor([0], device=multi_turn_response_mask.device)], dim=0
        )
        diff_end = torch.diff(mask_appended)
        ends = torch.where(diff_end == -1)[0] - 1
        segments = []
        for s, e in zip(starts, ends):
            segments.append(input_ids[s : e + 1].tolist())

        # Decode each segment
        # decoded_responses = [tokenizer.decode(seg, skip_special_tokens=True) for seg in segments]
        decoded_responses = tokenizer.batch_decode(segments, skip_special_tokens=True)
        return decoded_responses

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        # shape: (B*R, response_length_total)
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            # Get valid prompt_ids w/o padding tokens
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            # Get valid response_ids w/o padding tokens
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            response_str = [response_str]
            # For multi turn, we maybe need `response_str` in a list format
            if 'multi_turn_response_mask' in data_item.batch:
                # `response_str` is a list now
                response_str = self.extract_responses_list(
                    self.tokenizer, data_item.batch['input_ids'], data_item.batch['multi_turn_response_mask']
                )

            # We need `ground_truth` to be a list to support multiple candidate answers
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth]
            if 'candidate_answers' in data_item.non_tensor_batch['reward_model']:
                candidate_answers = data_item.non_tensor_batch['reward_model']['candidate_answers']
                if isinstance(candidate_answers, list):
                    ground_truth += candidate_answers
                elif isinstance(candidate_answers, str):
                    ground_truth += json.loads(candidate_answers)
                else:
                    raise TypeError(f"candidate_answers must be a list or a string, but got {type(candidate_answers)}")
            ground_truth = [g for g in ground_truth if isinstance(g, str)]
            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)

        return reward_tensor
