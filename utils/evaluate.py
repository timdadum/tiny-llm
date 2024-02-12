def clean_bpe_output(text):
    # From GPT:
    # Replace the BPE's word-start marker with a space
    # Note: This might introduce extra spaces in some cases, which we'll trim later
    text = text.replace(' ', '')
    
    # Additional common cleanup for BPE and similar tokenizations
    # This list can be extended based on specific characters you encounter
    replacements = {
        # Commonly seen in outputs, replace with appropriate character or remove
        'Ġ': ' ',  # A specific sequence that might appear, adjust as needed
        'Ã': '',    # Remove or replace based on your observation
        'â': '',    # Examples of characters that might need handling
        'ĺ': '',    # Remove or replace based on your observation
        'Ċ': '',    # Remove or replace
        'ċ': '',    # Remove or replace
        'š': '',    # Remove or replace
        'č': '',    # Remove or replace
        'ģ': '',    # Remove or replace
        # Add more replacements as needed based on the output you observe
    }
    
    # Perform replacements
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Post-cleanup: trim extra spaces introduced by replacements, if any
    text = ' '.join(text.split())
    
    return text