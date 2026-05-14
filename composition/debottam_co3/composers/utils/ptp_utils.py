import abc

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Union, Tuple, List
import torch.nn.functional as F


def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def find_subsequence_indices(tokens, phrase_tokens):
    #todo: Currently only returns the first occurrence of the phrase in the tokens.
    for i in range(len(tokens) - len(phrase_tokens) + 1):
        if tokens[i:i+len(phrase_tokens)] == phrase_tokens:
            return list(range(i, i + len(phrase_tokens)))
    raise ValueError(f"Phrase {' '.join(phrase_tokens)} not found in tokens: {' '.join(tokens)}")


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

def get_concept_indices(tokenizer, prompt, concepts):
    concept_indices_list = []
    index_offset = 1
    input_ids = tokenize_prompt(tokenizer, prompt)[0]  # shape: (77,)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    tokens = [t for t in tokens if t not in tokenizer.all_special_tokens and not t.startswith("<|")]
    tokens = [t.lower().strip("Ġ").strip("##") for t in tokens]  # Clean up for both CLIP/BERT styles

    for i, concept in enumerate(concepts):
        concept_ids = tokenize_prompt(tokenizer, concept)[0]
        concept_tokens = tokenizer.convert_ids_to_tokens(concept_ids)
        concept_tokens = [t for t in concept_tokens if t not in tokenizer.all_special_tokens and not t.startswith("<|")]
        concept_tokens = [t.lower().strip("Ġ").strip("##") for t in concept_tokens]
        concept_indices = find_subsequence_indices(tokens, concept_tokens) # Todo: only returns the first match, need to handle multiple matches
        concept_indices = [i + index_offset for i in concept_indices]  # Adjust indices to account for the first non-informative tokens
        concept_indices_list.append(concept_indices)

    return concept_indices_list


def get_all_noun_chunks(text, parser):
    doc = parser(text)
    noun_chunks = []
    for sentence in doc.sentences:
        words = sentence.words
        for word in words:
            # A noun phrase head is often a NOUN or PROPN
            if word.upos in ("NOUN", "PROPN"):
                # Collect modifiers and the head noun
                chunk_tokens = []
                for w in words:
                    if (
                        w.head == word.id and w.deprel in ("amod", "compound", "det")
                    ) or w.id == word.id:
                        chunk_tokens.append(w.text)
                # Sort by token order
                chunk_tokens = sorted(chunk_tokens, key=lambda t: text.index(t))
                chunk_text = " ".join(chunk_tokens)
                if chunk_text not in noun_chunks:
                    noun_chunks.append(chunk_text)

    return noun_chunks

def get_non_head_noun_chunks(text, parser):
    """
    warning: head nouns are NOUNs before 1st
    """
    doc = parser(text)
    non_head_chunks = []

    for sentence in doc.sentences:
        # Find the main syntactic head noun of the sentence (if any)
        head_noun_id = None
        for w in sentence.words:
            if w.head == 0 and w.upos in ("NOUN", "PROPN"):
                head_noun_id = w.id
                break

        # Now extract noun chunks excluding that head noun
        for w in sentence.words:
            if w.upos in ("NOUN", "PROPN"):
                # Skip if this is the sentence head noun
                if w.id == head_noun_id:
                    continue

                # Collect modifiers + head noun
                chunk_tokens = []
                for ww in sentence.words:
                    if (
                        ww.head == w.id and ww.deprel in ("amod", "compound", "det")
                    ) or ww.id == w.id:
                        chunk_tokens.append(ww)

                # Sort tokens by their position in sentence
                chunk_tokens = sorted(chunk_tokens, key=lambda t: t.id)
                chunk_text = " ".join(t.text for t in chunk_tokens)

                if chunk_text not in non_head_chunks:
                    non_head_chunks.append(chunk_text)
    
    return non_head_chunks


def remove_adjectives(text, parser):
    if isinstance(text, list):
        for i in range(len(text)):
            doc = parser(text[i])
            # Check if it's Stanza (has .sentences) or spaCy (direct iteration)
            if hasattr(doc, 'sentences'):
                # Stanza format
                text[i] = " ".join([token.text for sent in doc.sentences for token in sent.words if token.upos != "ADJ"])
            else:
                # spaCy format
                text[i] = " ".join([token.text for token in doc if token.pos_ != "ADJ"])
    else:
        doc = parser(text)
        # Check if it's Stanza (has .sentences) or spaCy (direct iteration)
        if hasattr(doc, 'sentences'):
            # Stanza format
            text = " ".join([token.text for sent in doc.sentences for token in sent.words if token.upos != "ADJ"])
        else:
            # spaCy format
            text = " ".join([token.text for token in doc if token.pos_ != "ADJ"])
    return text

def remove_articles_from_beginning(text, parser):
    if isinstance(text, list):
        for i in range(len(text)):
            doc = parser(text[i])
            # Check if it's Stanza (has .sentences) or spaCy (direct iteration)
            if hasattr(doc, 'sentences'):
                # Stanza format
                tokens = [token.text for sent in doc.sentences for token in sent.words]
                pos_tags = [token.upos for sent in doc.sentences for token in sent.words]
                
                # Remove articles from beginning
                start_idx = 0
                while start_idx < len(pos_tags) and pos_tags[start_idx] == "DET":
                    start_idx += 1
                
                text[i] = " ".join(tokens[start_idx:])
            else:
                # spaCy format
                tokens = [token.text for token in doc]
                pos_tags = [token.pos_ for token in doc]
                
                # Remove articles from beginning
                start_idx = 0
                while start_idx < len(pos_tags) and pos_tags[start_idx] == "DET":
                    start_idx += 1
                
                text[i] = " ".join(tokens[start_idx:])
    else:
        doc = parser(text)
        # Check if it's Stanza (has .sentences) or spaCy (direct iteration)
        if hasattr(doc, 'sentences'):
            # Stanza format
            tokens = [token.text for sent in doc.sentences for token in sent.words]
            pos_tags = [token.upos for sent in doc.sentences for token in sent.words]
            
            # Remove articles from beginning
            start_idx = 0
            while start_idx < len(pos_tags) and pos_tags[start_idx] == "DET":
                start_idx += 1
            
            text = " ".join(tokens[start_idx:])
        else:
            # spaCy format
            tokens = [token.text for token in doc]
            pos_tags = [token.pos_ for token in doc]
            
            # Remove articles from beginning
            start_idx = 0
            while start_idx < len(pos_tags) and pos_tags[start_idx] == "DET":
                start_idx += 1
            
            text = " ".join(tokens[start_idx:])
    
    return text

def remove_articles(text, parser):
    if isinstance(text, list):
        for i in range(len(text)):
            doc = parser(text[i])
            # Check if it's Stanza (has .sentences) or spaCy (direct iteration)
            if hasattr(doc, 'sentences'):
                # Stanza format
                text[i] = " ".join([token.text for sent in doc.sentences for token in sent.words if token.upos != "DET"])
            else:
                # spaCy format
                text[i] = " ".join([token.text for token in doc if token.pos_ != "DET"])
    else:
        doc = parser(text)
        # Check if it's Stanza (has .sentences) or spaCy (direct iteration)
        if hasattr(doc, 'sentences'):
            # Stanza format
            text = " ".join([token.text for sent in doc.sentences for token in sent.words if token.upos != "DET"])
        else:
            # spaCy format
            text = " ".join([token.text for token in doc if token.pos_ != "DET"])
    return text

def remove_conjunctions(text, parser):
    if isinstance(text, list):
        for i in range(len(text)):
            doc = parser(text[i])
            # Check if it's Stanza (has .sentences) or spaCy (direct iteration)
            if hasattr(doc, 'sentences'):
                # Stanza format
                text[i] = " ".join([token.text for sent in doc.sentences for token in sent.words if token.upos != "CCONJ"])
            else:
                # spaCy format
                text[i] = " ".join([token.text for token in doc if token.pos_ != "CCONJ"])
    else:
        doc = parser(text)
        # Check if it's Stanza (has .sentences) or spaCy (direct iteration)
        if hasattr(doc, 'sentences'):
            # Stanza format
            text = " ".join([token.text for sent in doc.sentences for token in sent.words if token.upos != "CCONJ"])
        else:
            # spaCy format
            text = " ".join([token.text for token in doc if token.pos_ != "CCONJ"])
    return text

def remove_conjunctions_from_beginning(text, parser):
    if isinstance(text, list):
        for i in range(len(text)):
            doc = parser(text[i])
            # Check if it's Stanza (has .sentences) or spaCy (direct iteration)
            if hasattr(doc, 'sentences'):
                # Stanza format
                tokens = [token.text for sent in doc.sentences for token in sent.words]
                pos_tags = [token.upos for sent in doc.sentences for token in sent.words]
                
                # Remove conjunctions from beginning
                start_idx = 0
                while start_idx < len(pos_tags) and pos_tags[start_idx] == "CCONJ":
                    start_idx += 1
                
                text[i] = " ".join(tokens[start_idx:])
            else:
                # spaCy format
                tokens = [token.text for token in doc]
                pos_tags = [token.pos_ for token in doc]
                
                # Remove conjunctions from beginning
                start_idx = 0
                while start_idx < len(pos_tags) and pos_tags[start_idx] == "CCONJ":
                    start_idx += 1
                
                text[i] = " ".join(tokens[start_idx:])
    else:
        doc = parser(text)
        # Check if it's Stanza (has .sentences) or spaCy (direct iteration)
        if hasattr(doc, 'sentences'):
            # Stanza format
            tokens = [token.text for sent in doc.sentences for token in sent.words]
            pos_tags = [token.upos for sent in doc.sentences for token in sent.words]
            
            # Remove conjunctions from beginning
            start_idx = 0
            while start_idx < len(pos_tags) and pos_tags[start_idx] == "CCONJ":
                start_idx += 1
            
            text = " ".join(tokens[start_idx:])
        else:
            # spaCy format
            tokens = [token.text for token in doc]
            pos_tags = [token.pos_ for token in doc]
            
            # Remove conjunctions from beginning
            start_idx = 0
            while start_idx < len(pos_tags) and pos_tags[start_idx] == "CCONJ":
                start_idx += 1
            
            text = " ".join(tokens[start_idx:])
    
    return text

def remove_wh_words(text, parser):
    excluded = ["who", "whom", "whose", "what", "which", "when", "where", "why"]
    if isinstance(text, list):
        for i in range(len(text)):
            doc = parser(text[i])
            # Check if it's Stanza (has .sentences) or spaCy (direct iteration)
            if hasattr(doc, 'sentences'):
                # Stanza format
                text[i] = " ".join([token.text for sent in doc.sentences for token in sent.words if token.text not in excluded])
            else:
                # spaCy format
                text[i] = " ".join([token.text for token in doc if token.text not in excluded])
            text[i] = text[i].strip()
    else:
        doc = parser(text)
        # Check if it's Stanza (has .sentences) or spaCy (direct iteration)
        if hasattr(doc, 'sentences'):
            # Stanza format
            text = " ".join([token.text for sent in doc.sentences for token in sent.words if token.text not in excluded])
        else:
            # spaCy format
            text = " ".join([token.text for token in doc if token.text not in excluded])
            text = text.strip()
        
    return text

def find_super_strings(st, chunks):

    super_strings = []
    
    for chunk in chunks:
        if st in chunk:
            super_strings.append(chunk)
    
    return super_strings

def find_sub_strings(st, chunks):

    sub_strings = []
    
    for chunk in chunks:
        if chunk in st:
            sub_strings.append(chunk)
    
    return sub_strings


def remove_duplicate_chunks(chunks):
    unique_chunks = []
    for chunk in chunks:
        if len(find_sub_strings(chunk, chunks)) == 1:
            unique_chunks.append(chunk)
    return unique_chunks

def get_prompts_and_concepts_fine(prompt, parser,
                             remove_adj_from_contrastive_prompts=False):
    
    non_head_noun_chunks = get_non_head_noun_chunks(prompt, parser)
    non_head_noun_chunks =  remove_duplicate_chunks(non_head_noun_chunks)
    non_head_noun_chunks = remove_articles(non_head_noun_chunks, parser)
    concepts = non_head_noun_chunks


    all_noun_chunks = get_all_noun_chunks(prompt, parser)
    prompts = remove_duplicate_chunks(all_noun_chunks)
    if remove_adj_from_contrastive_prompts:
        prompts = remove_adjectives(prompts, parser)
    return prompts, concepts

def get_prompts_and_concepts_coarse(prompt, parser,
                             remove_adj_from_contrastive_prompts=False,
                             remove_art_from_contrastive_prompts=False,
                             remove_art_from_concepts=True):

    concepts = get_all_noun_chunks(prompt, parser)
    concepts =  remove_duplicate_chunks(concepts)
    if remove_art_from_concepts:
        concepts = remove_articles(concepts, parser)

    all_noun_chunks = get_all_noun_chunks(prompt, parser)
    prompts = remove_duplicate_chunks(all_noun_chunks)
    if remove_adj_from_contrastive_prompts:
        prompts = remove_adjectives(prompts, parser)
    if remove_art_from_contrastive_prompts:
        prompts = remove_articles(prompts, parser)
    return prompts, concepts