import nltk
from nltk.corpus import wordnet as wn
from caltech101 import Caltech101

# download WordNet data
nltk.download('wordnet')
nltk.download('omw-1.4')


dataset = Caltech101(root="~/datasets/", download=True)

print(dataset.categories)
print(len(dataset.categories))
print(len(dataset))

i = 5000
first_example = dataset[i]
label_name = dataset.categories[first_example[1]]

print("First example:", first_example)
print("label:", label_name)


##################################
# WORDNET

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
        return [(clean_label, {clean_label})]  # Return original if no synsets found
    
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
hierarchy = get_wordnet_hierarchy(label_name)
print("WordNet hierarchy (specific to general) with synonyms and generality:")
for level, (concept, synonyms, depth, hyponym_count) in enumerate(hierarchy):
    print(f"  Level {level}: {concept} (depth: {depth}, hyponyms: {hyponym_count})")
    print(f"    Synonyms: {', '.join(synonyms)}")

#first_example[0].show()

##################################
# ConceptNet

import requests
import json
from collections import defaultdict

def get_conceptnet_hierarchy(label):
    """Get ConceptNet hierarchy for a given label with synonyms and generality measures"""
    # Clean the label
    clean_label = label.lower().replace('_', ' ')
    
    try:
        # Get ConceptNet concepts and relationships
        concept_uri = f"/c/en/{clean_label.replace(' ', '_')}"
        hierarchy = build_conceptnet_hierarchy(concept_uri)
        
        return hierarchy
        
    except Exception as e:
        print(f"Error accessing ConceptNet: {e}")
        return [(clean_label, {clean_label}, 0, 0)]

def query_conceptnet(uri, relation=None, limit=100):
    """Query ConceptNet API for a given URI"""
    base_url = "http://api.conceptnet.io"
    params = {
        'limit': limit,
        'filter': '/c/en'  # Only English concepts
    }
    
    if relation:
        params['rel'] = relation
    
    url = f"{base_url}{uri}"
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def get_conceptnet_synonyms(concept_uri):
    """Get synonyms for a ConceptNet concept"""
    synonyms = set()
    
    # Query for synonym relationships
    data = query_conceptnet(concept_uri, relation='/r/Synonym')
    
    for edge in data.get('edges', []):
        start_uri = edge['start']['@id']
        end_uri = edge['end']['@id']
        
        # Extract concept name from URI
        if start_uri == concept_uri and '/c/en/' in end_uri:
            synonym = end_uri.split('/c/en/')[-1].replace('_', ' ')
            synonyms.add(synonym)
        elif end_uri == concept_uri and '/c/en/' in start_uri:
            synonym = start_uri.split('/c/en/')[-1].replace('_', ' ')
            synonyms.add(synonym)
    
    # Add the original concept
    if '/c/en/' in concept_uri:
        original = concept_uri.split('/c/en/')[-1].replace('_', ' ')
        synonyms.add(original)
    
    return synonyms

def get_conceptnet_hyponyms_count(concept_uri):
    """Get count of hyponyms (more specific concepts) for a ConceptNet concept"""
    # Query for hyponym relationships (concepts that are instances/types of this one)
    relations = ['/r/IsA', '/r/InstanceOf']
    hyponym_count = 0
    
    for relation in relations:
        data = query_conceptnet(concept_uri, relation=relation)
        # Count edges where this concept is the end (more general concept)
        for edge in data.get('edges', []):
            if edge['end']['@id'] == concept_uri:
                hyponym_count += 1
    
    return hyponym_count

def build_conceptnet_hierarchy(concept_uri):
    """Build hierarchy from ConceptNet concept following IsA/InstanceOf relations"""
    hierarchy = []
    visited = set()
    current_uri = concept_uri
    depth = 0
    
    while current_uri and current_uri not in visited and depth < 10:
        visited.add(current_uri)
        
        # Extract concept name
        concept_name = current_uri.split('/c/en/')[-1].replace('_', ' ') if '/c/en/' in current_uri else current_uri
        
        # Get synonyms
        synonyms = get_conceptnet_synonyms(current_uri)
        
        # Get hyponym count
        hyponym_count = get_conceptnet_hyponyms_count(current_uri)
        
        hierarchy.append((concept_name, synonyms, depth, hyponym_count))
        
        # Find hypernym (more general concept) for next iteration
        data = query_conceptnet(current_uri, relation='/r/IsA')
        
        next_uri = None
        for edge in data.get('edges', []):
            if edge['start']['@id'] == current_uri and '/c/en/' in edge['end']['@id']:
                next_uri = edge['end']['@id']
                break
        
        current_uri = next_uri
        depth += 1
    
    return hierarchy

# Get ConceptNet hierarchy
print("\n" + "="*50)
concept_hierarchy = get_conceptnet_hierarchy(label_name)
print("ConceptNet hierarchy (specific to general) with synonyms and generality:")
for level, (concept, synonyms, depth, hyponym_count) in enumerate(concept_hierarchy):
    print(f"  Level {level}: {concept} (depth: {depth}, hyponyms: {hyponym_count})")
    print(f"    Synonyms: {', '.join(list(synonyms)[:5])}{'...' if len(synonyms) > 5 else ''}")

##################################
# Wikidata

import requests
import json
from typing import Dict, List, Set, Tuple

def get_wikidata_hierarchy(label):
    """Get Wikidata hierarchy for a given label with synonyms and generality measures"""
    # Clean the label
    clean_label = label.lower().replace('_', ' ')
    
    try:
        # Search for Wikidata entity
        entity_id = search_wikidata_entity(clean_label)
        if not entity_id:
            return [(clean_label, {clean_label}, 0, 0)]
        
        # Build hierarchy
        hierarchy = build_wikidata_hierarchy(entity_id)
        return hierarchy
        
    except Exception as e:
        print(f"Error accessing Wikidata: {e}")
        return [(clean_label, {clean_label}, 0, 0)]

def search_wikidata_entity(term):
    """Search for Wikidata entity ID for a given term"""
    url = "https://www.wikidata.org/w/api.php"
    params = {
        'action': 'wbsearchentities',
        'search': term,
        'language': 'en',
        'format': 'json',
        'limit': 1
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    if data.get('search'):
        return data['search'][0]['id']
    return None

def get_wikidata_entity(entity_id):
    """Get Wikidata entity information"""
    url = "https://www.wikidata.org/w/api.php"
    params = {
        'action': 'wbgetentities',
        'ids': entity_id,
        'format': 'json',
        'languages': 'en'
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    return data.get('entities', {}).get(entity_id, {})

def get_wikidata_labels_and_aliases(entity_data):
    """Extract labels and aliases from Wikidata entity"""
    labels = set()
    
    # Main label
    if 'labels' in entity_data and 'en' in entity_data['labels']:
        labels.add(entity_data['labels']['en']['value'])
    
    # Aliases
    if 'aliases' in entity_data and 'en' in entity_data['aliases']:
        for alias in entity_data['aliases']['en']:
            labels.add(alias['value'])
    
    return labels

def get_wikidata_subclass_count(entity_id):
    """Get count of subclasses for a Wikidata entity using SPARQL"""
    sparql_url = "https://query.wikidata.org/sparql"
    query = f"""
    SELECT (COUNT(?subclass) as ?count) WHERE {{
        ?subclass wdt:P279 wd:{entity_id} .
    }}
    """
    
    params = {
        'query': query,
        'format': 'json'
    }
    
    try:
        response = requests.get(sparql_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        bindings = data.get('results', {}).get('bindings', [])
        if bindings:
            return int(bindings[0]['count']['value'])
    except:
        pass
    
    return 0

def build_wikidata_hierarchy(entity_id):
    """Build hierarchy from Wikidata entity following subclass relationships"""
    hierarchy = []
    visited = set()
    current_id = entity_id
    depth = 0
    
    while current_id and current_id not in visited and depth < 15:
        visited.add(current_id)
        
        # Get entity data
        entity_data = get_wikidata_entity(current_id)
        if not entity_data:
            break
        
        # Extract concept name and synonyms
        labels = get_wikidata_labels_and_aliases(entity_data)
        concept_name = list(labels)[0] if labels else current_id
        
        # Get subclass count
        subclass_count = get_wikidata_subclass_count(current_id)
        
        hierarchy.append((concept_name, labels, depth, subclass_count))
        
        # Find superclass (P279: subclass of) for next iteration
        next_id = None
        if 'claims' in entity_data and 'P279' in entity_data['claims']:
            for claim in entity_data['claims']['P279']:
                if 'mainsnak' in claim and 'datavalue' in claim['mainsnak']:
                    next_id = claim['mainsnak']['datavalue']['value']['id']
                    break
        
        current_id = next_id
        depth += 1
    
    return hierarchy

# Get Wikidata hierarchy
print("\n" + "="*50)
wikidata_hierarchy = get_wikidata_hierarchy(label_name)
print("Wikidata hierarchy (specific to general) with synonyms and generality:")
for level, (concept, synonyms, depth, subclass_count) in enumerate(wikidata_hierarchy):
    print(f"  Level {level}: {concept} (depth: {depth}, subclasses: {subclass_count})")
    print(f"    Synonyms/Aliases: {', '.join(list(synonyms)[:5])}{'...' if len(synonyms) > 5 else ''}")

##################################
# UBY

import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Set, Tuple

def get_uby_hierarchy(label):
    """Get UBY hierarchy for a given label with synonyms and generality measures"""
    # Clean the label
    clean_label = label.lower().replace('_', ' ')
    
    try:
        # Search for UBY lexical entry
        lexical_entries = search_uby_lexical_entries(clean_label)
        if not lexical_entries:
            return [(clean_label, {clean_label}, 0, 0)]
        
        # Use the first lexical entry
        entry_id = lexical_entries[0]['id']
        
        # Build hierarchy
        hierarchy = build_uby_hierarchy(entry_id)
        return hierarchy
        
    except Exception as e:
        print(f"Error accessing UBY: {e}")
        return [(clean_label, {clean_label}, 0, 0)]

def search_uby_lexical_entries(term):
    """Search for UBY lexical entries for a given term"""
    # UBY API endpoint (hypothetical - actual endpoint may vary)
    url = "https://uby.ukp.informatik.tu-darmstadt.de/uby/api/lexicalEntries"
    params = {
        'lemma': term,
        'language': 'en',
        'pos': 'noun',
        'format': 'xml'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        entries = []
        
        for entry in root.findall('.//LexicalEntry'):
            entry_id = entry.get('id')
            lemma = entry.find('.//Lemma')
            if lemma is not None:
                entries.append({
                    'id': entry_id,
                    'lemma': lemma.get('writtenForm', term)
                })
        
        return entries
        
    except Exception as e:
        # Fallback if UBY API is not available
        print(f"UBY API not accessible: {e}")
        return []

def get_uby_synset_info(synset_id):
    """Get UBY synset information"""
    url = f"https://uby.ukp.informatik.tu-darmstadt.de/uby/api/synsets/{synset_id}"
    params = {'format': 'xml'}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        
        # Extract synonyms from synset
        synonyms = set()
        for sense in root.findall('.//Sense'):
            lemma_elem = sense.find('.//Lemma')
            if lemma_elem is not None:
                synonyms.add(lemma_elem.get('writtenForm', ''))
        
        return {
            'id': synset_id,
            'synonyms': synonyms,
            'definition': root.find('.//Definition')
        }
        
    except Exception as e:
        return {'id': synset_id, 'synonyms': set(), 'definition': None}

def get_uby_semantic_relations(entry_id):
    """Get semantic relations for a UBY lexical entry"""
    url = f"https://uby.ukp.informatik.tu-darmstadt.de/uby/api/semanticRelations"
    params = {
        'sourceId': entry_id,
        'format': 'xml'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        relations = []
        
        for relation in root.findall('.//SemanticRelation'):
            rel_type = relation.get('relType')
            target_id = relation.get('target')
            relations.append({
                'type': rel_type,
                'target': target_id
            })
        
        return relations
        
    except Exception as e:
        return []

def get_uby_hyponym_count(synset_id):
    """Get count of hyponyms for a UBY synset"""
    url = f"https://uby.ukp.informatik.tu-darmstadt.de/uby/api/semanticRelations"
    params = {
        'targetId': synset_id,
        'relType': 'hyponym',
        'format': 'xml'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        return len(root.findall('.//SemanticRelation'))
        
    except Exception as e:
        return 0

def build_uby_hierarchy(entry_id):
    """Build hierarchy from UBY lexical entry following hypernym relations"""
    hierarchy = []
    visited = set()
    current_id = entry_id
    depth = 0
    
    while current_id and current_id not in visited and depth < 15:
        visited.add(current_id)
        
        # Get synset information (assuming entry is linked to synset)
        synset_info = get_uby_synset_info(current_id)
        concept_name = list(synset_info['synonyms'])[0] if synset_info['synonyms'] else current_id
        synonyms = synset_info['synonyms']
        
        # Get hyponym count
        hyponym_count = get_uby_hyponym_count(current_id)
        
        hierarchy.append((concept_name, synonyms, depth, hyponym_count))
        
        # Find hypernym relation for next iteration
        relations = get_uby_semantic_relations(current_id)
        next_id = None
        
        for relation in relations:
            if relation['type'] in ['hypernym', 'broader']:
                next_id = relation['target']
                break
        
        current_id = next_id
        depth += 1
    
    return hierarchy

def get_uby_hierarchy_fallback(label):
    """Fallback UBY hierarchy using simulated data when API is not available"""
    clean_label = label.lower().replace('_', ' ')
    
    # Simulated UBY-style hierarchy for demonstration
    # In practice, this would use actual UBY lexical resources
    hierarchy = [
        (clean_label, {clean_label}, 0, 5),  # Original concept
        (f"{clean_label}_category", {f"{clean_label}_category", "category"}, 1, 15),  # Category level
        ("entity", {"entity", "thing", "object"}, 2, 50),  # Top level
    ]
    
    return hierarchy

# Get UBY hierarchy (with fallback)
print("\n" + "="*50)
try:
    uby_hierarchy = get_uby_hierarchy(label_name)
    if len(uby_hierarchy) == 1 and uby_hierarchy[0][2] == 0:  # Only original term returned
        uby_hierarchy = get_uby_hierarchy_fallback(label_name)
        print("Using UBY fallback hierarchy (API not available):")
    else:
        print("UBY hierarchy (specific to general) with synonyms and generality:")
except:
    uby_hierarchy = get_uby_hierarchy_fallback(label_name)
    print("Using UBY fallback hierarchy (API not available):")

for level, (concept, synonyms, depth, hyponym_count) in enumerate(uby_hierarchy):
    print(f"  Level {level}: {concept} (depth: {depth}, hyponyms: {hyponym_count})")
    print(f"    Synonyms: {', '.join(list(synonyms)[:5])}{'...' if len(synonyms) > 5 else ''}")

##################################
# Tree of Life (Taxonomy)

import requests
import json
from typing import Dict, List, Set, Tuple

def get_tree_of_life_hierarchy(label):
    """Get Tree of Life (biological taxonomy) hierarchy for a given label"""
    # Clean the label
    clean_label = label.lower().replace('_', ' ')
    
    try:
        # Search for taxonomic information
        taxon_info = search_tree_of_life(clean_label)
        if not taxon_info:
            return [(clean_label, {clean_label}, 0, 0)]
        
        # Build taxonomic hierarchy
        hierarchy = build_taxonomic_hierarchy(taxon_info)
        return hierarchy
        
    except Exception as e:
        print(f"Error accessing Tree of Life: {e}")
        return [(clean_label, {clean_label}, 0, 0)]

def search_tree_of_life(term):
    """Search for taxonomic information using GBIF API"""
    # GBIF Species API
    url = "https://api.gbif.org/v1/species/suggest"
    params = {
        'q': term,
        'limit': 1
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data:
            species_key = data[0].get('key')
            if species_key:
                return get_gbif_species_details(species_key)
                
    except Exception as e:
        # Fallback to OpenTree of Life API
        return search_opentree(term)
    
    return None

def get_gbif_species_details(species_key):
    """Get detailed species information from GBIF"""
    url = f"https://api.gbif.org/v1/species/{species_key}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except:
        return None

def search_opentree(term):
    """Search OpenTree of Life API as fallback"""
    url = "https://api.opentreeoflife.org/v3/tnrs/match_names"
    data = {
        "names": [term]
    }
    
    try:
        response = requests.post(url, json=data, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        if result.get('results'):
            return result['results'][0]
    except:
        pass
    
    return None

def build_taxonomic_hierarchy(taxon_info):
    """Build taxonomic hierarchy from species information"""
    hierarchy = []
    
    # Standard taxonomic ranks in order from specific to general
    taxonomic_ranks = [
        'species', 'genus', 'family', 'order', 
        'class', 'phylum', 'kingdom', 'domain'
    ]
    
    depth = 0
    for rank in taxonomic_ranks:
        if rank in taxon_info:
            taxon_name = taxon_info[rank]
            if taxon_name:
                # Get synonyms (scientific names, common names)
                synonyms = get_taxonomic_synonyms(taxon_name, rank)
                
                # Estimate subcategory count based on rank
                subcategory_count = estimate_taxonomic_subcategories(rank)
                
                hierarchy.append((taxon_name, synonyms, depth, subcategory_count))
                depth += 1
    
    return hierarchy

def get_taxonomic_synonyms(taxon_name, rank):
    """Get synonyms for a taxonomic name"""
    synonyms = {taxon_name}
    
    # Try to get additional names from GBIF
    try:
        url = "https://api.gbif.org/v1/species/suggest"
        params = {'q': taxon_name, 'limit': 5}
        
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            for result in data:
                if result.get('canonicalName'):
                    synonyms.add(result['canonicalName'])
                if result.get('vernacularName'):
                    synonyms.add(result['vernacularName'])
    except:
        pass
    
    return synonyms

def estimate_taxonomic_subcategories(rank):
    """Estimate number of subcategories for a taxonomic rank"""
    # Rough estimates based on known biodiversity
    estimates = {
        'species': 0,
        'genus': 10,
        'family': 100,
        'order': 1000,
        'class': 5000,
        'phylum': 20000,
        'kingdom': 100000,
        'domain': 1000000
    }
    
    return estimates.get(rank, 0)

def get_tree_of_life_fallback(label):
    """Fallback taxonomic hierarchy when APIs are not available"""
    clean_label = label.lower().replace('_', ' ')
    
    # Try to infer if it's a biological term
    if any(term in clean_label for term in ['animal', 'plant', 'bird', 'fish', 'insect', 'mammal']):
        hierarchy = [
            (clean_label, {clean_label}, 0, 0),
            ("Genus_unknown", {"genus", "unknown genus"}, 1, 10),
            ("Family_unknown", {"family", "unknown family"}, 2, 100),
            ("Order_unknown", {"order", "unknown order"}, 3, 1000),
            ("Class_unknown", {"class", "unknown class"}, 4, 5000),
            ("Phylum_unknown", {"phylum", "unknown phylum"}, 5, 20000),
            ("Animalia", {"Animalia", "animals", "kingdom animalia"}, 6, 100000),
            ("Eukaryota", {"Eukaryota", "eukaryotes", "domain eukaryota"}, 7, 1000000)
        ]
    else:
        hierarchy = [
            (clean_label, {clean_label}, 0, 0),
            ("Unknown taxonomy", {"unknown"}, 1, 0)
        ]
    
    return hierarchy

# Get Tree of Life hierarchy
print("\n" + "="*50)
try:
    tol_hierarchy = get_tree_of_life_hierarchy(label_name)
    if len(tol_hierarchy) == 1 and tol_hierarchy[0][2] == 0:
        tol_hierarchy = get_tree_of_life_fallback(label_name)
        print("Using Tree of Life fallback hierarchy (APIs not available):")
    else:
        print("Tree of Life hierarchy (specific to general) with taxonomic information:")
except:
    tol_hierarchy = get_tree_of_life_fallback(label_name)
    print("Using Tree of Life fallback hierarchy (APIs not available):")

for level, (concept, synonyms, depth, subcategory_count) in enumerate(tol_hierarchy):
    print(f"  Level {level}: {concept} (depth: {depth}, subcategories: {subcategory_count})")
    print(f"    Names/Synonyms: {', '.join(list(synonyms)[:5])}{'...' if len(synonyms) > 5 else ''}")

##################################
# DBpedia Ontology

import requests
import json
from typing import Dict, List, Set, Tuple
from SPARQLWrapper import SPARQLWrapper, JSON

def get_dbpedia_hierarchy(label):
    """Get DBpedia Ontology hierarchy for a given label with synonyms and generality measures"""
    # Clean the label
    clean_label = label.lower().replace('_', ' ')
    
    try:
        # Search for DBpedia resource
        resource_uri = search_dbpedia_resource(clean_label)
        if not resource_uri:
            return [(clean_label, {clean_label}, 0, 0)]
        
        # Build ontology hierarchy
        hierarchy = build_dbpedia_hierarchy(resource_uri)
        return hierarchy
        
    except Exception as e:
        print(f"Error accessing DBpedia: {e}")
        return [(clean_label, {clean_label}, 0, 0)]

def search_dbpedia_resource(term):
    """Search for DBpedia resource using DBpedia Lookup service"""
    url = "https://lookup.dbpedia.org/api/search"
    params = {
        'query': term,
        'format': 'json',
        'maxResults': 1
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('docs'):
            return data['docs'][0]['resource'][0]
            
    except Exception as e:
        # Fallback to direct URI construction
        clean_term = term.replace(' ', '_').title()
        return f"http://dbpedia.org/resource/{clean_term}"
    
    return None

def query_dbpedia_sparql(query):
    """Execute SPARQL query against DBpedia endpoint"""
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    try:
        results = sparql.query().convert()
        return results['results']['bindings']
    except Exception as e:
        print(f"SPARQL query failed: {e}")
        return []

def get_dbpedia_types_and_hierarchy(resource_uri):
    """Get DBpedia types and build hierarchy"""
    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT DISTINCT ?type ?label ?superClass WHERE {{
        <{resource_uri}> rdf:type ?type .
        ?type rdfs:label ?label .
        OPTIONAL {{ ?type rdfs:subClassOf ?superClass }}
        FILTER (lang(?label) = "en")
        FILTER (STRSTARTS(str(?type), "http://dbpedia.org/ontology/"))
    }}
    """
    
    return query_dbpedia_sparql(query)

def get_dbpedia_superclasses(class_uri):
    """Get superclasses for a DBpedia ontology class"""
    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT DISTINCT ?superClass ?label WHERE {{
        <{class_uri}> rdfs:subClassOf* ?superClass .
        ?superClass rdfs:label ?label .
        FILTER (lang(?label) = "en")
        FILTER (STRSTARTS(str(?superClass), "http://dbpedia.org/ontology/"))
    }}
    ORDER BY ?superClass
    """
    
    return query_dbpedia_sparql(query)

def get_dbpedia_subclass_count(class_uri):
    """Get count of subclasses for a DBpedia ontology class"""
    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT (COUNT(DISTINCT ?subClass) as ?count) WHERE {{
        ?subClass rdfs:subClassOf <{class_uri}> .
        FILTER (STRSTARTS(str(?subClass), "http://dbpedia.org/ontology/"))
    }}
    """
    
    results = query_dbpedia_sparql(query)
    if results:
        return int(results[0]['count']['value'])
    return 0

def get_dbpedia_synonyms(resource_uri):
    """Get synonyms and alternative names from DBpedia"""
    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX dbp: <http://dbpedia.org/property/>
    
    SELECT DISTINCT ?name WHERE {{
        <{resource_uri}> rdfs:label|dbo:alias|dbp:name ?name .
        FILTER (lang(?name) = "en")
    }}
    """
    
    results = query_dbpedia_sparql(query)
    synonyms = set()
    for result in results:
        synonyms.add(result['name']['value'])
    
    return synonyms

def build_dbpedia_hierarchy(resource_uri):
    """Build hierarchy from DBpedia resource following ontology classes"""
    hierarchy = []
    
    # Get types for the resource
    types_data = get_dbpedia_types_and_hierarchy(resource_uri)
    
    if not types_data:
        # Fallback - just return the resource name
        resource_name = resource_uri.split('/')[-1].replace('_', ' ')
        return [(resource_name, {resource_name}, 0, 0)]
    
    # Find the most specific type (usually the first one that's not too general)
    main_type = None
    for type_info in types_data:
        type_uri = type_info['type']['value']
        if 'Thing' not in type_uri and 'Agent' not in type_uri:
            main_type = type_uri
            break
    
    if not main_type and types_data:
        main_type = types_data[0]['type']['value']
    
    if main_type:
        # Get the class hierarchy
        superclasses = get_dbpedia_superclasses(main_type)
        
        # Build hierarchy from specific to general
        visited = set()
        depth = 0
        
        # Start with the main type
        hierarchy.append(build_dbpedia_class_entry(main_type, depth))
        visited.add(main_type)
        depth += 1
        
        # Add superclasses
        for superclass_info in superclasses:
            superclass_uri = superclass_info['superClass']['value']
            if superclass_uri not in visited and 'Thing' not in superclass_uri:
                hierarchy.append(build_dbpedia_class_entry(superclass_uri, depth))
                visited.add(superclass_uri)
                depth += 1
                
                if depth > 10:  # Limit hierarchy depth
                    break
    
    return hierarchy

def build_dbpedia_class_entry(class_uri, depth):
    """Build a single hierarchy entry for a DBpedia class"""
    class_name = class_uri.split('/')[-1]
    
    # Get label
    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?label WHERE {{
        <{class_uri}> rdfs:label ?label .
        FILTER (lang(?label) = "en")
    }}
    """
    
    results = query_dbpedia_sparql(query)
    if results:
        class_name = results[0]['label']['value']
    
    # Get synonyms (alternative labels)
    synonyms = {class_name}
    
    # Get subclass count
    subclass_count = get_dbpedia_subclass_count(class_uri)
    
    return (class_name, synonyms, depth, subclass_count)

def get_dbpedia_hierarchy_fallback(label):
    """Fallback DBpedia hierarchy when SPARQL queries fail"""
    clean_label = label.lower().replace('_', ' ')
    
    # Simple ontology-based hierarchy
    hierarchy = [
        (clean_label, {clean_label}, 0, 0),
        ("Physical Object", {"Physical Object", "Object"}, 1, 100),
        ("Thing", {"Thing", "Entity"}, 2, 1000)
    ]
    
    return hierarchy

# Get DBpedia hierarchy
print("\n" + "="*50)
try:
    # Check if SPARQLWrapper is available
    from SPARQLWrapper import SPARQLWrapper, JSON
    
    dbpedia_hierarchy = get_dbpedia_hierarchy(label_name)
    if len(dbpedia_hierarchy) == 1 and dbpedia_hierarchy[0][2] == 0:
        dbpedia_hierarchy = get_dbpedia_hierarchy_fallback(label_name)
        print("Using DBpedia fallback hierarchy (SPARQL not available):")
    else:
        print("DBpedia Ontology hierarchy (specific to general):")
except ImportError:
    print("SPARQLWrapper not installed. Install with: pip install SPARQLWrapper")
    dbpedia_hierarchy = get_dbpedia_hierarchy_fallback(label_name)
    print("Using DBpedia fallback hierarchy:")
except:
    dbpedia_hierarchy = get_dbpedia_hierarchy_fallback(label_name)
    print("Using DBpedia fallback hierarchy (API not available):")

for level, (concept, synonyms, depth, subclass_count) in enumerate(dbpedia_hierarchy):
    print(f"  Level {level}: {concept} (depth: {depth}, subclasses: {subclass_count})")
    print(f"    Synonyms/Labels: {', '.join(list(synonyms)[:5])}{'...' if len(synonyms) > 5 else ''}")

