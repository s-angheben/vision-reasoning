import nltk
from nltk.corpus import wordnet as wn
from caltech101 import Caltech101

# download WordNet data
nltk.download('wordnet')
nltk.download('omw-1.4')


dataset = Caltech101(root="~/datasets/", download=True)

print(dataset.categories)
print(len(dataset))

def get_wordnet_hierarchy(label):
    """Get WordNet hierarchy for a given label with synonyms and generality measures"""
    # Clean the label - remove underscores, make lowercase
    clean_label = label.lower().replace('_', ' ')
    
    # Try to find synsets for the label
    synsets = wn.synsets(clean_label, pos=wn.NOUN)
    
    if not synsets:
        # Try individual words if compound word fails
        words = clean_label.split()
        for word in words:
            synsets = wn.synsets(word, pos=wn.NOUN)
            if synsets:
                break
    
    if not synsets:
        return [(clean_label, {clean_label}, 0, 0)]  # Return original with 4 values
    
    # Use the first synset (most common meaning)
    synset = synsets[0]
    
    # Get hypernym path with synonyms and generality measures
    hierarchy = []
    for i, hypernym in enumerate(synset.hypernym_paths()[0]):
        concept_name = hypernym.name().split('.')[0].replace('_', ' ')
        synonyms = set(lemma.replace('_', ' ') for lemma in hypernym.lemma_names())
        
        # Generality measures
        depth = i  # Level in hierarchy (0 = most specific)
        hyponym_count = len(list(hypernym.closure(lambda s: s.hyponyms())))  # Number of subcategories
        
        hierarchy.append((concept_name, synonyms, depth, hyponym_count))
    
    return hierarchy


# Get WordNet hierarchy
for label in dataset.categories:
    print(f"Label: {label}")
    hierarchy = get_wordnet_hierarchy(label)
    print("WordNet hierarchy (specific to general):")
    for level, item in enumerate(hierarchy):
        # Handle both old format (2 values) and new format (4 values)
        if len(item) == 4:
            concept, synonyms, depth, hyponym_count = item
            print(f"  Level {level}: {concept} (depth: {depth}, hyponyms: {hyponym_count})")
            print(f"    Synonyms: {', '.join(synonyms)}")
        else:
            concept, synonyms = item
            print(f"  Level {level}: {concept}")
            print(f"    Synonyms: {', '.join(synonyms)}")
    print("\n")
