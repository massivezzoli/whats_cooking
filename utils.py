import re
from nltk.stem import WordNetLemmatizer

#### Preprocessing functions:
# Remove ingredients that appear only once in whole dataset
def remove_ing(input_data, ingr_rm):
    input = [x for x in input_data if x not in ingr_rm]
    return input
# Cleaning steps and tokenization
def preprocess_ing(input_data, single_word=False):
    mod_recipe = []
    for ingredient in input_data:
        mod_ingredient = []
        for word in ingredient.split():
            #remove anything between parenthesis, including par.
            word = re.sub(r'\([^)]*\)', '', word)
            # remove anything that is not a letter
            word = re.sub('[^a-zA-Z]', '', word)
            # lower case, lemmatize and strip extra chars
            mod_ingredient.append(WordNetLemmatizer().lemmatize(word.lower().strip(",.!:?;'u'\u00AE'u'\u2122' ")))
        if single_word:
            #put word in alphabetic order
            mod_ingredient.sort()
            mod_recipe.append('_'.join(mod_ingredient))
        else:
            mod_recipe.append(' '.join(mod_ingredient))
    return ' '.join(mod_recipe)
