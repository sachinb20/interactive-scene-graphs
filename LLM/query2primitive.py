from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
import numpy as np  

# device="cuda"

# translation_lm_id = 'stsb-roberta-large'
# translation_lm = SentenceTransformer(translation_lm_id).to(device)

# action_list = ["navigate to the apple","pick the apple","place the apple in sink"]
# action_list_embedding = translation_lm.encode(action_list, batch_size=512, convert_to_tensor=True, device=device)  # lower batch_size if limited by GPU memory

# # helper function for finding similar sentence in a corpus given a query
# def find_most_similar(query_str, corpus_embedding):
#     query_embedding = translation_lm.encode(query_str, convert_to_tensor=True, device=device)
#     # calculate cosine similarity against each candidate sentence in the corpus
#     cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
#     # retrieve high-ranked index and similarity score
#     most_similar_idx, matching_score = np.argmax(cos_scores), np.max(cos_scores)
#     return most_similar_idx, matching_score

# most_similar_idx, matching_score = find_most_similar("navigate to the apple", action_list_embedding)
# print(action_list[most_similar_idx],matching_score)


class SentenceSimilarity:
    def __init__(self, model_id='stsb-roberta-large', device='cuda'):
        self.device = device
        self.model = SentenceTransformer(model_id).to(self.device)
        
    def encode_sentences(self, sentences, batch_size=512):
        return self.model.encode(sentences, batch_size=batch_size, convert_to_tensor=True, device=self.device)
    
    def find_most_similar(self, query_str, corpus_embedding):
        query_embedding = self.model.encode(query_str, convert_to_tensor=True, device=self.device)
        cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
        most_similar_idx, matching_score = np.argmax(cos_scores), np.max(cos_scores)
        return most_similar_idx, matching_score

# Example usage
action_list = ["navigate to the apple", "pick the apple", "place the apple in sink"]
sentence_similarity = SentenceSimilarity()

action_list_embedding = sentence_similarity.encode_sentences(action_list)
most_similar_idx, matching_score = sentence_similarity.find_most_similar("navigate to the apple", action_list_embedding)

print(action_list[most_similar_idx], matching_score)
