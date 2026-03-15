import re
labels = [
    'Yamun Angcadan 2008 memorial with flowers and blue banner',
    'man giving speech at podium with flowers',
    'President Duterte giving speech in front of podium with logo',
    'colorful tulips and daffodils in garden',
    'woman in traditional Malay outfit with orange scarf and white skirt',
    'colorful flowers',
    'man',
    'flower',
    'loc_858>flower',
    'footwear'
]

def clean_and_filter(all_labels):
    cleaned_labels = []
    for label in all_labels:
        # clean loc tags
        label = re.sub(r'(?:<loc_\d+>|loc_\d+>)', '', label).strip()
        
        # clean prefixes
        cleaned = label.strip()
        lower_cleaned = cleaned.lower()
        for prefix in ["a ", "an ", "the "]:
            if lower_cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                lower_cleaned = cleaned.lower()
                break
        
        # valid check
        if not cleaned:
            continue
            
        # exclude special chars that are not normal punctuation
        if re.search(r'[^a-zA-Z0-9\s\-\,\.\']', cleaned):
            continue
            
        if len(cleaned) < 2:
            continue
            
        if len(cleaned.split()) > 8:
            continue
            
        cleaned_labels.append(cleaned)
    return cleaned_labels

print(clean_and_filter(labels))
