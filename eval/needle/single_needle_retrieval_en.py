import json
import sys
import os
import random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_context, insert_needle

def load_english_needles(dataset_path):
    """
    Load all English needles from NeedleBench needles.jsonl dataset.

    Args:
        dataset_path (str): Path to needles.jsonl.

    Returns:
        list of dict: Filtered list with keys: needle, retrieval_question, gold_standard_answer
    """
    needles = []
    with open(dataset_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get('language') == 'English':
                needles.append({
                    'needle': entry.get('needle', ''),
                    'retrieval_question': entry.get('retrieval_question', ''),
                    'gold_standard_answer': entry.get('gold_standard_answer', '')
                })
    print(f"Loaded {len(needles)} English needles from {dataset_path}")
    return needles


def find_needle_token_indices(needle, prompt, tokenizer, model):
    # tokenization and find needle token indices
    needle_start_char = prompt.find(needle[:20])
    needle_end_char = needle_start_char + len(needle)
    # print(f"Needle starts at char {needle_start_char} and ends at char {needle_end_char} in the prompt.")

    # Check how the model performs
    prompt = tokenizer(prompt, return_offsets_mapping = True, return_tensors="pt")
    offsets = prompt["offset_mapping"][0]  # offsets of each token in the prompt
    input_ids = prompt['input_ids'].to(model.device)
    attn_mask = prompt["attention_mask"].to(model.device)

    # print("After tokenization, there is %d tokens" % len(input_ids[0]))

    # find token indices whose offsets overlap with the needle (standard token to char alignment in HuggingFace)
    needle_token_indices = []
    for idx, (start, end) in enumerate(offsets.tolist()):
        if end > needle_start_char and start < needle_end_char:
            needle_token_indices.append(idx)
    # print("Needle token indices:", needle_token_indices)
    return input_ids, attn_mask, needle_token_indices


def niah_insert_needle_wrapper(context, needle, question,tokenizer, model, depth=0.5):
    """
    Insert the needle into the context at a specified depth.
    The depth determines how many characters from the start of the context are preserved.
    """
    if depth < 0 or depth > 1:
        raise ValueError("Depth must be between 0 and 1.")
    
    context = insert_needle(context, needle, depth=depth)
    prompt ="\n<|im_start|> This is a very long story book: <book> %s </book>.\n" % context
    prompt += "Based on the content of the book, Question: %s\nAnswer:" % question
    input_ids, attn_mask, needle_token_indices = find_needle_token_indices(needle, prompt, tokenizer, model)

    return input_ids, attn_mask, needle_token_indices

def sample_needle_and_construct_prompt(needles_list, ctx_len, tokenizer,model, depth=0.5):
    """
    Sample a needle from the list and construct prompt with context.

    Args:
        needles_list (list of dict): List of needles with retrieval questions.
        ctx_len (int): Context length.
        depth (float): Insertion depth.

    Returns:
        tuple: (needle, context, retrieval_question, input_ids, attn_mask, needle_token_indices)
    """
    # Sample one needle entry (or iterate systematically in your loop)
    sample = random.choice(needles_list)
    needle = sample['needle']
    question = sample['retrieval_question']
    gold_standard_answer = sample['gold_standard_answer']

    # Load context
    context = load_context(fpath="eval/needle/PaulGrahamEssays/*.txt", ctx_len=ctx_len)

    # Insert needle and get token indices
    input_ids, attn_mask, needle_token_indices = niah_insert_needle_wrapper(context, needle, question, tokenizer, model, depth=depth)

    return needle, context, question, gold_standard_answer, input_ids, attn_mask, needle_token_indices




