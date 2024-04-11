# Created using ChatGPT (GPT-4), 11-4-2024
# Prompt: 'Could you generate me a small debug text that has a vocabulary of about 128 words, and is about 8192 words long?'

# Define a simple vocabulary of 128 unique words
vocabulary = [
    'the', 'village', 'was', 'surrounded', 'by', 'forest', 'and', 'river', 'people', 'lived',
    'in', 'harmony', 'with', 'nature', 'every', 'morning', 'birds', 'sang', 'trees', 'whispered',
    'stories', 'of', 'old', 'children', 'played', 'fields', 'while', 'adults', 'worked', 'soil',
    'gathered', 'food', 'from', 'woods', 'fish', 'swam', 'waters', 'clear', 'as', 'crystal',
    'sun', 'rose', 'above', 'mountains', 'bathing', 'everything', 'golden', 'light', 'evening',
    'families', 'came', 'together', 'share', 'their', 'day', 'stories', 'laughter', 'filled',
    'air', 'night', 'fell', 'stars', 'appeared', 'sky', 'like', 'diamonds', 'on', 'a', 'velvet',
    'cloth', 'moon', 'cast', 'gentle', 'glow', 'over', 'land', 'peace', 'reigned', 'until',
    'one', 'day', 'stranger', 'came', 'bringing', 'news', 'world', 'beyond', 'villagers',
    'listened', 'intently', 'tales', 'adventure', 'discovery', 'decided', 'that', 'would',
    'explore', 'learn', 'more', 'about', 'this', 'vast', 'unknown', 'set', 'out', 'dawn',
    'journey', 'began'
]

# Sample text template using the vocabulary
text_template = [
    'The village was surrounded by a forest and a river. People lived in harmony with nature. ',
    'Every morning, the birds sang and the trees whispered stories of old. Children played in the fields, ',
    'while the adults worked the soil and gathered food from the woods. The fish swam in waters clear as crystal. ',
    'The sun rose above the mountains, bathing everything in golden light. In the evening, families came together ',
    'to share their day\'s stories. Laughter filled the air. When night fell, the stars appeared in the sky like diamonds on a velvet cloth. ',
    'The moon cast a gentle glow over the land, and peace reigned. Until one day, a stranger came, bringing news of the world beyond. ',
    'The villagers listened intently to tales of adventure and discovery. They decided that they too would explore and learn more about this vast unknown. ',
    'As the first light of dawn touched the sky, their journey began.'
]

# Generate the text by repeating the template to reach approximately 8192 words
total_words_target = 8192
current_word_count = sum([len(sentence.split()) for sentence in text_template])
repetitions_needed = total_words_target // current_word_count

# Generate the debug text
debug_text = ' '.join(text_template * repetitions_needed)

print(debug_text)