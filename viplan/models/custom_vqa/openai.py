import base64
from collections import defaultdict
from io import BytesIO
from typing import List, Optional

import openai
from openai.types.chat.chat_completion import ChatCompletion
import numpy as np
from PIL.Image import Image

OPENAI_MODEL_ID_PREFIX = "gpt"


class OpenAIVQA:
    def __init__(self, model_id: str, system_prompt: Optional[str], api_key: str, **inference_kwargs):
        assert model_id.startswith(OPENAI_MODEL_ID_PREFIX), (
            f"Model ID {model_id} not supported. "
            f"Only OpenAI models starting with prefix {OPENAI_MODEL_ID_PREFIX} are supported."
        )

        self.model_id = model_id
        self.system_prompt = system_prompt
        self.inference_kwargs = inference_kwargs

        # initialize OpenAI client.
        # we use this to make calls to the OpenAI API.
        self.openai_client = openai.OpenAI(api_key=api_key)

    def __call__(self, images: List[Image], query_batch: List[str], *token_groups_of_interest: List[List[str]]) -> List[List[float]]:
        # OpenAI API does not support batching with separate
        # logits outputs per query, so we process each query separately.
        # This should still be efficient due to internal optimizations
        # and caching within the OpenAI API.
        batch_responses = [
            self.estimation_query(
                images=images,
                query=query,
                **self.inference_kwargs,
            )
            for query in query_batch
        ]

        # Extract token group probabilities for each response.
        # Example: if token_groups_of_interest = (["yes", "Yes"], ["no", "No"]),
        # then for each response we get [P(yes), P(no)].
        # we get one such list per query in the batch.
        return [
            self.extract_token_group_probs(resp, *token_groups_of_interest)
            for resp in batch_responses
        ]

    def extract_token_group_probs(self, model_output: ChatCompletion, *token_groups: List[List[str]]) -> List[float]:
        # get top logprobs for the first generated token.
        # OpenAI API supports up to 20 top logprobs to avoid model distilation.
        top_logprobs = model_output.choices[0].logprobs.content[0].top_logprobs

        # map tokens to their probabilities
        tok_to_prob = defaultdict(int)
        for item in top_logprobs:
            tok_to_prob[item.token] += np.exp(item.logprob)

        # get probs for tokens of interest
        token_groups_probs = [
            np.sum([tok_to_prob[token] for token in token_group])
            for token_group in token_groups
        ]

        # normalize each group vs all groups
        return [
            prob / np.sum(token_groups_probs, axis=0) for prob in token_groups_probs
        ]

    def estimation_query(
        self,
        images: List[Image],
        query: str,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> ChatCompletion:
        # convert images to base64 strings
        processed_images = [_preprocess_image(image) for image in images]

        # construct image input as per OpenAI API spec
        images_as_urls = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
            for base64_image in processed_images
        ]

        # construct user role message
        input_message_content = [{"type": "text", "text": query}] + images_as_urls
        input_message = [
            {"role": "user", "content": input_message_content},
        ]

        # add system prompt if provided
        if self.system_prompt is not None:
            input_message = [{"role": "developer", "content": self.system_prompt}] + input_message

        # make the API call to OpenAI
        response = self.openai_client.chat.completions.create(
            messages=input_message,
            model=self.model_id,
            max_completion_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=True,
            top_logprobs=20,
        )

        return response


def _preprocess_image(image: Image) -> str:
    # Create a BytesIO object
    buffered = BytesIO()

    # Save the image to the BytesIO stream in JPEG format

    image.convert("RGB").save(buffered, format="JPEG")

    # Retrieve the byte data from the BytesIO stream
    img_bytes = buffered.getvalue()

    # Encode the byte data to Base64
    img_base64 = base64.b64encode(img_bytes)

    # Decode the Base64 bytes to a string
    return img_base64.decode("utf-8")
